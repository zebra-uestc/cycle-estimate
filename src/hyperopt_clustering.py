from hyperopt import hp, fmin, Trials, space_eval
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import os
import umap
import hyperopt
from file_handler import read_bbv, read_encoded_bbv, write_encoded_bbv
from meta import root_dir
import cluster
import utils

BBV_data = []
Benchmark = ""

def hyper_opt(benchmark, epoch=200):
    print("=================   Hyperopt   =================")
    global BBV_data 
    global Benchmark
    Benchmark = benchmark
    # read bbv and cycle from .txt file
    data_dir = f"{root_dir}/data/{benchmark}"
    bbv_file = f"{data_dir}/bbv.txt"
    bbv_data = read_bbv(bbv_file)

    # normalize bbv data
    scaler = MinMaxScaler()
    bbv_data = scaler.fit_transform(bbv_data)
    BBV_data = bbv_data
    # define the hyperparameter search space
    space = {
        'hdbscan__min_cluster_size': hp.choice('hdbscan__min_cluster_size', np.arange(100, 200, 10)),
        'hdbscan__cluster_selection_epsilon': hp.choice('hdbscan__cluster_selection_epsilon', [0.1, 0.5, 0.9, 2.0, 3.0, 5.0, 8, 10, 15]),
        'hdbscan2__min_cluster_size': hp.choice('hdbscan2__min_cluster_size', [2, 3, 4, 5]),
        'umap__n_neighbors': hp.choice('umap__n_neighbors', np.arange(50, 70, 20)),
        'umap__min_dist': hp.choice('umap__min_dist', [0.1, 0.5]),
        'umap__n_components': hp.choice('umap__n_components', [8, 10, 12, 14, 15]),
    }
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=hyperopt.rand.suggest, max_evals=epoch, trials=trials)
    best_params = space_eval(space, best)
    return best_params
                            
def objective(params):
    print(params)
    global BBV_data 
    global Benchmark
    pipeline = Pipeline([
        ('umap',
            umap.UMAP(
                n_neighbors=params['umap__n_neighbors'],
                min_dist=params['umap__min_dist'],
                n_components=params['umap__n_components'],
            )),
        ('hdbscan',
            hdbscan.HDBSCAN(min_cluster_size=params['hdbscan__min_cluster_size'],
                            cluster_selection_epsilon=params['hdbscan__cluster_selection_epsilon'],
            ))
    ])
    # evaluate the current parameter set
    score = evaluate_clustering(pipeline, BBV_data, [
        params['umap__n_neighbors'], params['umap__min_dist'], params['umap__n_components'], params['hdbscan__min_cluster_size'],
        params['hdbscan__cluster_selection_epsilon'], params['hdbscan2__min_cluster_size']
    ], Benchmark)
    return score


def evaluate_clustering(pipeline, data, params, benchmark):
    data_dir = f"{root_dir}/data/{benchmark}"
    files_in_dir = os.listdir(data_dir)
    file_name = 'encoder_' + str(params[0]) + '_' + str(params[1]) + '_' + str(params[2]) + '.txt'
    file_path = data_dir + '/' + file_name
    if file_name in files_in_dir:
        data_transformed = read_encoded_bbv(file_path)
    else:
        data_transformed = pipeline.named_steps['umap'].fit_transform(data)
        write_encoded_bbv(data_transformed, file_path)
    cluster_labels = pipeline.named_steps['hdbscan'].fit_predict(data_transformed)
    _, _, max_k = cluster.assign_cluster(data_transformed, cluster_labels, benchmark, params[5])
    error, _ = utils.calc_error(benchmark)
    content = f'{error}% {max_k} {params}'
    file_path = f"{root_dir}/data/{benchmark}/hyper_errors.txt"
    with open(file_path, 'a') as file:
        file.write(content)
    return utils.calc_score(error, max_k)


def dump_result(params, benchmark):
    data_dir = f"{root_dir}/data/{benchmark}"
    # dump the bbv_encoded data(dim-reduced bbv) corresponding to the best parameters
    file_name = 'encoder_' + str(params['umap__n_neighbors']) + '_' + str(params['umap__min_dist']) + '_' + str(
        params['umap__n_components']) + '.txt'
    file_path = data_dir + '/' + file_name
    bbv_encoded = read_encoded_bbv(file_path)
    write_encoded_bbv(bbv_encoded, f"{data_dir}/bbv_encoded.txt")

    cluster_labels = hdbscan.HDBSCAN(
        min_cluster_size=params['hdbscan__min_cluster_size'],
        cluster_selection_epsilon=params['hdbscan__cluster_selection_epsilon']).fit_predict(bbv_encoded)

    ori_max, ori_err, max_k = cluster.assign_cluster(bbv_encoded, cluster_labels, benchmark, params['hdbscan2__min_cluster_size'], flag=True)
    final_err, speedup = utils.calc_error(benchmark)
    return ori_max, ori_err, max_k, final_err, speedup
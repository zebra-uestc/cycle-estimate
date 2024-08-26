from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import hdbscan
import numpy as np
import os
import umap

from file_handler import read_bbv, read_encoded_bbv, write_encoded_bbv
from meta import root_dir
import cluster
import utils


# def grid_search(benchmark, epoch=1):
#     # read bbv and cycle from .txt file
#     data_dir = f"{root_dir}/data/{benchmark}"
#     bbv_file = f"{data_dir}/bbv.txt"
#     bbv_data = read_bbv(bbv_file)

#     # normalize bbv data
#     scaler = MinMaxScaler()
#     bbv_data = scaler.fit_transform(bbv_data)

#     # create dim_reduce and cluster pipeline
#     dim_reducer = umap.UMAP()
#     clusterer = hdbscan.HDBSCAN()
#     pipeline = Pipeline([('umap', dim_reducer), ('hdbscan', clusterer)])

#     # setup a parameter grid
#     param_grid = {
#         'umap__n_neighbors': np.arange(50, 160, 20),
#         'umap__min_dist': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
#         'umap__n_components': [9, 10, 11, 12, 13, 14, 15],
#         'hdbscan__min_cluster_size': np.arange(50, 160, 20),
#         'hdbscan__cluster_selection_epsilon': [0.1, 0.5, 0.9, 2.0, 3.0, 5.0, 6.0],
#         'hdbscan2__min_cluster_size': [2, 3, 4, 5, 6, 7, 8, 9],
#     }

#     # search for the best set of parameters
#     best_score = np.inf  # the smaller is better
#     best_params = {
#         'umap__n_neighbors': 50,
#         'umap__min_dist': 0.1,
#         'umap__n_components': 8,
#         'hdbscan__min_cluster_size': 50,
#         'hdbscan__cluster_selection_epsilon': 0.1,
#         'hdbscan2__min_cluster_size': 2,
#     }
#     for umap_n_neighbors in param_grid['umap__n_neighbors']:
#         for umap_min_dist in param_grid['umap__min_dist']:
#             for umap_n_components in param_grid['umap__n_components']:
#                 for hdbscan_min_cluster_size in param_grid['hdbscan__min_cluster_size']:
#                     if bbv_data.shape[0] > 20000:
#                         if hdbscan_min_cluster_size < 100:
#                             continue
#                     for hdbscan_cluster_selection_epsilon in param_grid['hdbscan__cluster_selection_epsilon']:
#                         for hdbscan2_min_cluster_size in param_grid['hdbscan2__min_cluster_size']:
#                             pipeline = Pipeline([
#                                 ('umap',
#                                  umap.UMAP(
#                                      n_neighbors=umap_n_neighbors,
#                                      min_dist=umap_min_dist,
#                                      n_components=umap_n_components,
#                                  )),
#                                 ('hdbscan',
#                                  hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
#                                                  cluster_selection_epsilon=hdbscan_cluster_selection_epsilon)),
#                             ])

#                             # evaluate the current parameter set
#                             score = evaluate_clustering(pipeline, bbv_data, [
#                                 umap_n_neighbors, umap_min_dist, umap_n_components, hdbscan_min_cluster_size,
#                                 hdbscan_cluster_selection_epsilon, hdbscan2_min_cluster_size
#                             ], benchmark)

#                             # update best_score and best_params
#                             if score < best_score:
#                                 best_score = score
#                                 best_params = {
#                                     'umap__n_neighbors': umap_n_neighbors,
#                                     'umap__min_dist': umap_min_dist,
#                                     'umap__n_components': umap_n_components,
#                                     'hdbscan__min_cluster_size': hdbscan_min_cluster_size,
#                                     'hdbscan__cluster_selection_epsilon': hdbscan_cluster_selection_epsilon,
#                                     'hdbscan2__min_cluster_size': hdbscan2_min_cluster_size,
#                                 }

#                             epoch -= 1
#                             if epoch == 0:
#                                 print(best_params)
#                                 return best_params


# def evaluate_clustering(pipeline, data, params, benchmark):
#     data_dir = f"{root_dir}/data/{benchmark}"
#     files_in_dir = os.listdir(data_dir)

#     file_name = 'encoder_' + str(params[0]) + '_' + str(params[1]) + '_' + str(params[2]) + '.txt'
#     file_path = data_dir + '/' + file_name

#     if file_name in files_in_dir:
#         data_transformed = read_encoded_bbv(file_path)
#     else:
#         data_transformed = pipeline.named_steps['umap'].fit_transform(data)
#         write_encoded_bbv(data_transformed, file_path)

#     cluster_labels = pipeline.named_steps['hdbscan'].fit_predict(data_transformed)

#     max_k = cluster.assign_cluster(data_transformed, cluster_labels, benchmark, params[5])
#     error, _ = utils.calc_error(benchmark)
#     return utils.calc_score(error, max_k)


# def dump_result(params, benchmark):
#     data_dir = f"{root_dir}/data/{benchmark}"

#     # dump the bbv_encoded data(dim-reduced bbv) corresponding to the best parameters
#     file_name = 'encoder_' + str(params['umap__n_neighbors']) + '_' + str(params['umap__min_dist']) + '_' + str(
#         params['umap__n_components']) + '.txt'
#     file_path = data_dir + '/' + file_name
#     bbv_encoded = read_encoded_bbv(file_path)
#     write_encoded_bbv(bbv_encoded, f"{data_dir}/bbv_encoded.txt")

#     # delete all intermediate bbv_encoded data
#     for file_name in os.listdir(data_dir):
#         if file_name.startswith("encoder") and file_name.endswith(".txt"):
#             file_path = os.path.join(data_dir, file_name)
#             os.remove(file_path)

#     cluster_labels = hdbscan.HDBSCAN(
#         min_cluster_size=params['hdbscan__min_cluster_size'],
#         cluster_selection_epsilon=params['hdbscan__cluster_selection_epsilon']).fit_predict(bbv_encoded)

#     cluster.assign_cluster(bbv_encoded, cluster_labels, benchmark, params['hdbscan2__min_cluster_size'], flag=True)
#     return utils.calc_error(benchmark)
 
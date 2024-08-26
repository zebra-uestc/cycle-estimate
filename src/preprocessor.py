from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from file_handler import *
from meta import root_dir
from hyperopt_clustering import dump_result
import hyperopt_clustering

def preprocess(benchmark):
    data_dir = f"{root_dir}/data/{benchmark}"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # search for the best set of parameters
    params = hyperopt_clustering.hyper_opt(benchmark)
    # dump the results data of dim-reduction and clustering
    ori_max, ori_err, max_k, err, speedup = dump_result(params, benchmark)
    print(f"no cali error: {err}")
    # read dim-reduced bbv
    bbv_encoded_file = f"{data_dir}/bbv_encoded.txt"
    bbv_encoded_data = read_encoded_bbv(bbv_encoded_file)

    # read cluster result
    simpoints0_file = f"{data_dir}/simpoints0"
    weights0_file = f"{data_dir}/weights0"
    label_file = f"{data_dir}/label.txt"
    id_map = read_simpoints0(simpoints0_file)
    weight_map = read_weights0(weights0_file)
    label_data = read_label(label_file)

    # read cycle data
    cycle_file = f"{data_dir}/cycle.txt"
    cycle_data = read_cycle(cycle_file)
    # calculate bbv_encoded_diff and cycle_diff
    center_indices = np.array([id_map[int(id), 1] for id in label_data])
    bbv_encoded_diff = bbv_encoded_data - bbv_encoded_data[center_indices, :]
    cycle_diff = cycle_data - cycle_data[center_indices]

    # normalize bbv_encoded_diff
    scaler = MinMaxScaler()
    bbv_diff_normalized = scaler.fit_transform(bbv_encoded_diff)

    X = bbv_diff_normalized
    y = cycle_diff

    # delete intermediate process file
    os.remove(f"{data_dir}/bbv_encoded.txt")

    # save useful data which will be used in regression
    np.save(f"{data_dir}/X.npy", X)
    np.save(f"{data_dir}/y.npy", y)

    print(f"preprocessing of the data for benchmark {benchmark} is complete.")
    print(f"-----cluster number {len(id_map)}------")
    return ori_max, ori_err, max_k, err, speedup

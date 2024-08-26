import numpy as np
from file_handler import *
from meta import root_dir
import nu_SVR

def select_best_method(benchmark):
    data_dir = f"{root_dir}/data/{benchmark}"

    # load data from .txt and .npy files
    imap = read_simpoints0(f"{data_dir}/simpoints0")
    wmap = read_weights0(f"{data_dir}/weights0")
    label = read_label(f"{data_dir}/label.txt")
    cycle = read_cycle(f"{data_dir}/cycle.txt")
    X = np.load(f"{data_dir}/X.npy")
    y = np.load(f"{data_dir}/y.npy")

    print("=================   Results   =================")
    # use numpy array to stage results
    result = np.empty((0, 4))

    # total_cycle_baseline
    inum = len(cycle)
    total_cycle_baseline = np.sum(cycle)
    print(f"total_cycle_baseline: {total_cycle_baseline}")

    # total_cycle_center (use cluster center representative for the wholw cluster)
    cluster_weight = wmap[:, 1]
    center_cycle = cycle[imap[:, 1]]
    total_cycle_center = int(np.sum(center_cycle * cluster_weight) * inum)
    center_error = np.abs(total_cycle_center - total_cycle_baseline) / total_cycle_baseline * 100
    entry = np.array(["cluster_center", total_cycle_baseline, total_cycle_center, f"{center_error:.6f}%"])
    result = np.vstack((result, entry))
    print("==> cluster_center:")
    print(f"total_cycle: {int(total_cycle_center)}, error: {center_error:.6f}%")
    np.savetxt(f"{data_dir}/center_cycle.txt", center_cycle, fmt="%d")

    # total_cycle_pred
    print("\n=============   Regression   =============")

    # for each interval，use label and imap to get the interval id of its cluster center.
    center_indices = np.array([imap[int(id), 1] for id in label])

    selected_X, selected_y = get_rdm_diff(label, X, y)
    nusvr_result = nu_SVR.optimize_nusvr(selected_X, selected_y, X, np.sum(cycle[center_indices]), total_cycle_baseline)
    print(f"=============opt nusvr_result {nusvr_result}==========")
    
    return nusvr_result, total_cycle_center

def get_rdm_diff(label, X, y):
    label_indices = {}
    for idx, label_value in enumerate(label):
        if label_value not in label_indices:
            label_indices[label_value] = []
        label_indices[label_value].append(idx)

    selected_indices = []
    selected_X = []
    selected_y = []
    sample_count = 1
    for label_value, indices in label_indices.items():
        sample_count = int(len(indices) * 0.005)
        if sample_count < 5:
            sample_count = 5
        if len(indices) >= sample_count:
            step_size = len(indices) // sample_count
            selected_indices.extend(indices[::step_size][:sample_count])
        else:
            selected_indices.extend(indices)
    selected_X = X[selected_indices]
    selected_y = y[selected_indices]
    return selected_X, selected_y

def process(benchmark):
    errors, cycle_center = select_best_method(benchmark)
    return errors

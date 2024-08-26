import numpy as np
from file_handler import read_cycle, read_simpoints0, read_weights0, read_encoded_bbv
from meta import root_dir


def calc_score(error, k):
    normalized_error = normalize(error, 0.0, 0.02)
    normalized_k = normalize(k, 8, 30)
    data_point = np.array([normalized_error, normalized_k])
    score = np.linalg.norm(data_point)
    return score


def calc_error(benchmark, write = False, args = None):
    data_dir = f"{root_dir}/data/{benchmark}"
    cycle_file = f"{data_dir}/cycle.txt"
    cycle_data = read_cycle(cycle_file)
    interval_num = len(cycle_data)
    total_cycle_baseline = np.sum(cycle_data)

    simpoints_file = f"{data_dir}/simpoints0"
    weights_file = f"{data_dir}/weights0"
    imap = read_simpoints0(simpoints_file)
    wmap = read_weights0(weights_file)
    center_cycle = cycle_data[imap[:, 1]]
    center_cycle_sum = np.sum(center_cycle)
    cluster_weight = wmap[:, 1]
    total_cycle_center = int(np.sum(center_cycle * cluster_weight) * interval_num)

    cycle_error = np.abs(total_cycle_center - total_cycle_baseline) / total_cycle_baseline * 100
    if write:
        output_file = f"{data_dir}/cpi_error_args.txt"
        with open(output_file, 'a') as outfile:
            outfile.write(f"{total_cycle_center} {cycle_error:.5f} {len(center_cycle)} {args}\n")
    return cycle_error, total_cycle_baseline/center_cycle_sum


def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


def save_res(file_name, data):
    with open(file_name, "w") as file:
        for line in data:
            file.write(line + "\n")


def update_label(benchmark, simpoints, labels):
    data_dir = f"{root_dir}/data/{benchmark}"
    bbv_encoded = read_encoded_bbv(f"{data_dir}/bbv_encoded.txt")
    center = []
    for idx in simpoints:
        i = idx[0]
        center.append(bbv_encoded[int(i)])
    center = np.array(center)
    for i in range(len(labels)):
        if int(labels[i]) == -1:
            distances = np.linalg.norm(center - bbv_encoded[i], axis=1)
            nearest_index = np.argmin(distances)
            labels[i] = nearest_index
    labels = np.array(labels)
    return labels


def write_result(res, labels, closest_points, benchmark, max_k, save_label=False):
    res = res.astype(int)
    total = 0
    for ri in range(max_k):
        total = res[ri] + total
    weight = []
    i = 0
    for i in range(max_k):
        if res[i] == 0:
            continue
        weight.append(round(res[i] / total, 6))
    simpoints = []
    index = 0
    center = closest_points
    for val in center:
        if val != -1:
            simpoints.append(f"{val} {index}")
            index += 1
    simpoints = np.array(simpoints)
    data_dir = f"{root_dir}/data/{benchmark}"
    output_file = data_dir + '/simpoints0'
    save_res(output_file, simpoints)
    weight_output_file = data_dir + '/weights0'
    weights0 = np.array([f"{val} {i}" for i, val in enumerate(weight)])
    save_res(weight_output_file, weights0)
    if save_label == True:
        labels = update_label(benchmark, simpoints, labels)
        labels_file = data_dir + '/label.txt'
        save_res(labels_file, labels.astype(str).tolist())

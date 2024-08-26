import numpy as np
import pandas as pd


def read_bbv(file_path):
    matrix = []
    with open(file_path, "r") as file:
        for line in file:
            row = line.split()
            bbs = []
            for item in row:
                bb = item.split(':')
                bb_id = int(bb[1])
                bb_insts = int(bb[2])
                bbs.append((bb_id, bb_insts))
            matrix.append(bbs)

    num_rows = len(matrix)
    num_cols = max(bbs[-1][0] for bbs in matrix)

    bbv_data = [[0] * num_cols for _ in range(num_rows)]
    for row_id, bbs in enumerate(matrix):
        for bb_id, bb_insts in bbs:
            bbv_data[row_id][bb_id - 1] = bb_insts
    return np.array(bbv_data)


def read_encoded_bbv(file_path):
    return np.loadtxt(file_path, delimiter=' ')


def read_simpoints0(file_path):
    map = []
    with open(file_path, 'r') as file:
        for line in file:
            item = line.split()
            key, val = int(item[1]), int(item[0])
            map.append([key, val])
    return np.array(map)


def read_weights0(file_path):
    map = []
    with open(file_path, 'r') as file:
        for line in file:
            item = line.split()
            key, val = int(item[1]), float(item[0])
            map.append([key, val])
    return np.array(map)


def read_label(file_path):
    with open(file_path, 'r') as file:
        label_data = [int(line.strip()) for line in file]
        return np.array(label_data)


def read_cycle(file_path):
    with open(file_path, 'r') as file:
        cycle_data = [int(line.strip()) for line in file]
        return np.array(cycle_data)


def write_encoded_bbv(bbv_encoded, file_path):
    np.savetxt(file_path, bbv_encoded, fmt="%.6f", delimiter=' ', newline='\n')


def write_xlsx(file_path, data, sheet_name='Sheet1'):
    df = pd.DataFrame(data, columns=["method", "total cycle baseline", "total cycle pred", "error"])

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Result written to {file_path} in sheet '{sheet_name}'")
    except FileNotFoundError:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Result written to {file_path} in sheet '{sheet_name}'")

def read_xlsx(file_path, sheet_name='Sheet1'):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Data read from {file_path} in sheet '{sheet_name}':")
        print(df)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")


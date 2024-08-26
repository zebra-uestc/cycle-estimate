import sys
import pandas as pd
from preprocessor import preprocess
from processor import process
from meta import root_dir

if __name__ == "__main__":
    benchmark = sys.argv[1]
    phase1_num, phase1_err, phase2_num, phase2_err, speedup = preprocess(benchmark)
    cali_error = process(benchmark)

    # 将数据写入 Excel 文件
    data = {
        "Phase1_Num": [phase1_num],
        "Phase1_Err": [phase1_err],
        "Phase2_Num": [phase2_num],
        "Phase2_Err": [phase2_err],
        "Speedup": [speedup],
        "Final_Error": [cali_error],
    }

    df = pd.DataFrame(data)  # 创建 DataFrame
    output_file = f"{root_dir}/data/{benchmark}/results.xlsx"  # 定义输出文件名
    df.to_excel(output_file, index=False, float_format="%.6f")  # 写入 Excel 文件
    print(f"Results saved to {output_file}")
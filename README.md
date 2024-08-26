**If you need to estimate the total number of clock cycles for the complete execution of a program under a new architecture**:
Please combine the clustering results from `simpoints0` and `weights0`, use NEMU to generate checkpoints at specified positions, and then use the aligned version of gem5 for the new architecture to test the number of clock cycles for executing the cluster center intervals. This will allow you to create a new `center_cycle.txt`.

For generating checkpoints with NEMU, please refer to [Generation and Execution of Checkpoints](https://docs.xiangshan.cc/zh-cn/latest/tools/simpoint/). For restoring checkpoints in gem5, please refer to [OpenXiangShan GEM5](https://github.com/OpenXiangShan/GEM5).

The specific run command is as follows:

```bash
# Replace your project path in the ./scripts/env.sh file
cd cycle-estimate
source ./scripts/env.sh
python src/main.py <benchmark_name>
```
The running results will be saved in the {root_dir}/data/{benchmark}/results.xlsx file：
Phase1_Num represents the number of clusters in the first stage clustering.
Phase1_Err represents the error of the first stage clustering.
Phase2_Num represents the number of clusters in the second stage clustering.
Phase2_Err represents the error of the second stage clustering.
Speedup represents the acceleration ratio between the final result and the full simulation.
Final_err represents the final result of SimPoint+.


**If you need to run other benchmarks that we haven't tested**, first obtain the simpoint_bbv.gz file through NEMU profiling, and decompress it to get `bbv.txt`. Then, run the benchmark once using the old architecture in gem5 to get the number of clock cycles for each interval, and write these values into the `cycle.txt` file. Finally, place bbv.txt and cycle.txt in the `data/<benchmark_name>` folder, and run the command as described above.

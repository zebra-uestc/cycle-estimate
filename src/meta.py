import os

# load project root dir from shell env
root_dir = os.getenv("CYCLE_ESTIMATE_DIR")

# it doesn't matter, you can set it to whatever value you want.
global_seed = 13

benchmarks = ["perlbench_checkspam", "bzip2_chicken", "gcc_166", "bwaves", "gamess_cytosine", "milc"]

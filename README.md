# 🌠 job_array_engine 🌠

<p align="center">
  <img src="logo/2x/logo.png" width="40%">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`job_array_engine` is a Python package that allows you to run a grid search on a cluster using SLURM. 

**Special ability**: it can fit *multiple grid search instances* in *one job*. Parameter: `--per_job [int]`.

It is tailored to work with the [AIMAGELAB](https://aimagelab.ing.unimore.it/) cluster, but it can be easily adapted to work with other clusters.

## Installation

You can install `job_array_engine` using pip:

```bash
pip install git+https://www.github.com/GianlucaMancusi/job_array_engine
```

## Usage

```python
from job_array_engine import GridSearchLauncher

launcher = GridSearchLauncher(
    job_name="JOB_NAME",
    n_gpus=1,
    slurm_output_path="PATH_TO_OUTPUT",
    python_file="PATH_TO_PYTHON_FILE.py",
    env_name="ENV_NAME",
    use_conda=True,
    project_path="PATH_TO_PROJECT_ROOT",
    mem=32,
    tot_time="00:30:00",
    cpu_per_task=2,
    per_job=1, # Number of commands to run in parallel in a single job
    exclude_nodes=['node1', 'node2'], # List of nodes to exclude
    env_vars={'WANDB__SERVICE_WAIT': 300}
)
launcher.set_grid_params(
    static_params="--arg1 val1 --arg2 val2",
    grid_params={
        "--arg3": ["val3", "val4"],
        "--arg4": [1, 2, 3],
    },
    flags=["--flag1", "--flag2"] # grid search on flags (optional)
)
launcher.generate_sbatch_script()
launcher.run_grid_search()
```

## License
[MIT](https://choosealicense.com/licenses/mit/) License


## Acknowledgements
This package was created by [Gianluca Mancusi](https://github.com/GianlucaMancusi) from a codebase made by [Aniello Panariello](https://github.com/apanariello4/). University of Modena and Reggio Emilia.

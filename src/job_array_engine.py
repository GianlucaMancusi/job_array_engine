"""
This script is used to launch a grid search on a slurm cluster.
It generates a bash script that will be launched with sbatch.
The bash script will launch a python script with different parameters.
The parameters are specified in the set_grid_params method.
"""

import math
import os
import subprocess
import time
import uuid

from pathlib import Path
import itertools


class GridSearchLauncher:
    """
    # `GridSearchLauncher` class

    This class is responsible for launching a grid search on a slurm cluster. It generates a bash script that will be launched with sbatch, which in turn will execute a python script with different parameters.

    ## `__init__` method

    Initializes an instance of the `GridSearchLauncher` class.

    **Args:**
    - `job_name` (str): The name of the job.
    - `n_gpus` (int): The number of GPUs.
    - `slurm_output_path` (str): The path to the slurm output.
    - `python_file` (str): The path to the python file.
    - `project_path` (str): The path to the project.
    - `mem` (int): The memory size in GB.
    - `tot_time` (str): The total time for the job.
    - `env_name` (str, optional): The name of the environment. Defaults to None.
    - `cpu_per_task` (int, optional): The number of CPUs per task. Defaults to 2.
    - `per_job` (int, optional): The number of jobs to run in parallel in a single node. Defaults to 1.
    - `exclude_nodes` (list, optional): The list of nodes to exclude. Defaults to None.
    - `partition` (str, optional): The partition name. Defaults is None.
    - `account` (str, optional): The account name. Defaults is None.
    - `use_conda` (bool, optional): Whether to use conda. Defaults to True.
    - `env_vars` (dict, optional): The environment variables. Defaults to None.

    ### Example: Basic Usage

    ```python
    launcher = GridSearchLauncher(
        job_name="JOB_NAME",
        n_gpus=1,
        slurm_output_path="PATH_TO_OUTPUT",
        python_file="PATH_TO_PYTHON_FILE.py",
        env_name="ENV_NAME",
        use_conda=True,
        project_path="./",
        mem=32,
        tot_time="00:30:00",
        cpu_per_task=2,
        per_job=1,
        exclude_nodes=['node1', 'node2'],
        env_vars={'WANDB__SERVICE_WAIT': 300}
    )
    launcher.set_grid_params(
        static_params="--arg1 val1 --arg2 val2",
        grid_params={
            "--arg3": ["val3", "val4"],
            "--arg4": [1, 2, 3],
        },
        flags=["--flag1", "--flag2"]
    )
    launcher.generate_sbatch_script()
    launcher.run_grid_search()
    """

    def __init__(
        self,
        job_name,
        n_gpus,
        slurm_output_path,
        python_file,
        project_path,
        mem,
        tot_time,
        env_name=None,
        cpu_per_task=2,
        per_job=1,
        exclude_nodes=None,
        partition=None,
        account=None,
        use_conda=True,
        env_vars=None,
    ):
        self.job_name = job_name
        self.n_gpus = n_gpus
        self.slurm_output_path = Path(slurm_output_path)
        self.python_file = Path(python_file)
        self.env_name = env_name
        self.project_path = Path(project_path)
        self.mem = mem
        self.tot_time = tot_time
        self.cpu_per_task = cpu_per_task
        self.per_job = per_job
        self.exclude_nodes = exclude_nodes or []
        self.partition = partition
        self.account = account
        self.use_conda = use_conda
        self.env_vars = env_vars or {}
        self.random_uuid = str(uuid.uuid4())[:4]
        self.static_params = ''
        self.grid_params = {}
        self.flags = ['']
        self.set_grid_params()

    def set_grid_params(self, static_params=None, grid_params=None, flags=None):
        """
        Set the grid search parameters
        :param static_params: string of static parameters (not to be grid-searched)
        :param grid_params: dictionary of parameters to grid-search (key=param_name, value=list_of_values)
        :param flags: list of flags to add to each job
        """
        self.static_params = static_params if static_params is not None else ''
        self.grid_params = grid_params if grid_params is not None else {}
        self.flags = flags if flags is not None else ['']

    def _generate_permutations(self):
        keys = list(self.grid_params.keys())
        values = list(self.grid_params.values())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def generate_sbatch_script(self):
        conf_list = [' '.join(f"{key} {value}" for key, value in config.items()) for config in self._generate_permutations()]
        conf_list = [f"{string} {flag}".strip() for string in conf_list for flag in self.flags]

        num_jobs = math.ceil(len(conf_list) / self.per_job)
        job_array = '' if num_jobs == 1 else f'#SBATCH --array=0-{num_jobs - 1}'
        array_file_name = '_%A' if num_jobs == 1 else "_%A_%a"

        srun_ = 'srun ' if self.per_job == 1 else ''
        arguments = "\n".join(f"'{srun_}python -u {self.python_file} {self.static_params} {config}'" for config in conf_list)

        job_string = "sleep $(($RANDOM % 20)); ${arguments[$SLURM_ARRAY_TASK_ID]}" if self.per_job == 1 else \
            ' &\nsleep 10s; '.join(f"${{arguments[$(($SLURM_ARRAY_TASK_ID * {self.per_job} + {i}))]}}" for i in range(self.per_job)) + ' &\nwait\n'

        env = f"\n. /usr/local/anaconda3/etc/profile.d/conda.sh\nconda activate {self.env_name}\n" if self.env_name and self.use_conda else \
              f"source activate {self.env_name}" if self.env_name else ''

        exclude_nodes_str = f"#SBATCH --exclude={','.join(self.exclude_nodes)}" if self.exclude_nodes else ""

        env_vars = "\n".join(f"export {key}={value}" for key, value in self.env_vars.items())

        recap = f"###>Recap: n_gpus={self.n_gpus}, mem={self.mem}G, time={self.tot_time}, env={self.env_name}, num_jobs={num_jobs}"

        output_sbatch = f"""#!/bin/bash
{recap}
{f'#SBATCH -p {self.partition}' if self.partition else ''}
{f'#SBATCH -A {self.account}' if self.account else ''}
#SBATCH --job-name={self.job_name}
#SBATCH --nodes=1
{job_array}
#SBATCH --output="{self.slurm_output_path}/{self.job_name}{array_file_name}.out"
#SBATCH --error="{self.slurm_output_path}/{self.job_name}{array_file_name}.err"
#SBATCH --time={self.tot_time}
#SBATCH --mem={self.mem}G
#SBATCH --gres=gpu:{self.n_gpus}
#SBATCH --cpus-per-task={self.cpu_per_task}
{exclude_nodes_str}
{env}
cd {self.project_path.resolve()}
export PYTHONPATH="{self.project_path.resolve()}"
{env_vars}

arguments=(
{arguments}
)

{job_string}
"""
        self.slurm_output_path.mkdir(parents=True, exist_ok=True)
        print(f"Generating sbatch files (n_jobs={num_jobs})")
        with open("__sbatch__.sh", "w") as f:
            f.write(output_sbatch)

    def run_grid_search(self):
        sbatch_file = self.project_path / "__sbatch__.sh"
        if not sbatch_file.exists():
            print("sbatch file (__sbatch__.sh) not found. Please run generate_sbatch_script()")
            return
        subprocess.run(f"cd {self.project_path}", shell=True)
        time.sleep(0.1)

        # open sbatch file, take second line, split on "###>", take second element
        recap = sbatch_file.read_text().splitlines()[1].split("###>")[1].strip()
        print(recap)

        print("\nLaunch sbatch? (Y/n)")
        if input() in ("y", "Y", ""):
            os.system(f"sbatch {sbatch_file}")


if __name__ == "__main__":
    launcher = GridSearchLauncher(
        job_name="JOB_NAME",
        n_gpus=1,
        slurm_output_path="PATH_TO_OUTPUT",
        python_file="PATH_TO_PYTHON_FILE.py",
        env_name="ENV_NAME",
        use_conda=True,
        project_path="./",
        mem=32,
        tot_time="00:30:00",
        cpu_per_task=2,
        per_job=1,  # Number of jobs to run in parallel in a single node
        exclude_nodes=['node1', 'node2'],  # List of nodes to exclude
        env_vars={'WANDB__SERVICE_WAIT': 300}
    )
    launcher.set_grid_params(
        static_params="--arg1 val1 --arg2 val2",
        grid_params={
            "--arg3": ["val3", "val4"],
            "--arg4": [1, 2, 3],
        },
        flags=["--flag1", "--flag2"]  # grid search on flags (optional)
    )
    launcher.generate_sbatch_script()
    launcher.run_grid_search()

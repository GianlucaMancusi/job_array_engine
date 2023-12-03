"""
This script is used to launch a grid search on a slurm cluster.
It generates a bash script that will be launched with sbatch.
The bash script will launch a python script with different parameters.
The parameters are specified in the set_grid_params method.
"""

import itertools
import math
import os
import signal
import subprocess
import time
import uuid


class GridSearchLauncher:
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
            partition="prod",
            use_conda=True,
    ):
        self.job_name = job_name
        self.n_gpus = n_gpus
        self.slurm_output_path = slurm_output_path
        self.python_file = python_file
        self.env_name = env_name
        self.project_path = project_path
        self.mem = mem
        self.tot_time = tot_time
        self.cpu_per_task = cpu_per_task
        self.per_job = per_job
        self.exclude_nodes = exclude_nodes if exclude_nodes is not None else []
        self.partition = partition
        self.use_conda = use_conda
        self.random_uuid = str(uuid.uuid4())[:4]
        self.static_params = None
        self.grid_params = None
        self.flags = None
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

        # generate all the possible combinations of parameters to grid search
        conf_list = []
        for config in self._generate_permutations():
            string = ''.join(f"{key} {value} " for key, value in config.items())
            conf_list.append(string[:-1])
        conf_list = [f"{string} {flag}" for string in conf_list for flag in self.flags]

        # number of jobs to launch (each job will launch per_job python scripts)
        num_jobs = math.ceil(len(conf_list) / self.per_job) 

        # Prepare sbatch file content
        job_array = '' if num_jobs == 1 else f'#SBATCH --array=0-{num_jobs - 1}'
        array_file_name = '_%A' if num_jobs == 1 else "_%A_%a"

        srun_ = 'srun ' if self.per_job == 1 else ''
        arguments = "\n".join(f"'{srun_}python -u {self.python_file} {self.static_params} {config}'" for config in conf_list)

        if self.per_job == 1:
            job_string = "sleep $(($RANDOM % 20)); ${arguments[$SLURM_ARRAY_TASK_ID]}"
        else:
            job_string = (
                ' &\nsleep 60s; '.join(
                    f"${{arguments[$(($SLURM_ARRAY_TASK_ID * {self.per_job} + {i}))]}}"
                    for i in range(self.per_job)
                )
                + ' &\nwait\n'
            )

        env = '' 
        if self.env_name:
            if self.use_conda:
                env = f"\n. /usr/local/anaconda3/etc/profile.d/conda.sh\nconda activate {self.env_name}\n"
            else:
                env = f"source activate {self.env_name}"

        output_sbatch = f"""#!/bin/bash
#SBATCH -p {self.partition}
#SBATCH --job-name={self.job_name}
#SBATCH --nodes=1
{job_array}
#SBATCH --output="{self.slurm_output_path}/{self.job_name}{array_file_name}.out"
#SBATCH --error="{self.slurm_output_path}/{self.job_name}{array_file_name}.err"
#SBATCH --time={self.tot_time}
#SBATCH --mem={self.mem}G
#SBATCH --gres=gpu:{self.n_gpus}
#SBATCH --cpus-per-task={self.cpu_per_task}
{f"#SBATCH --exclude={','.join(self.exclude_nodes)}" if self.exclude_nodes else ""}
{env}
cd {self.project_path}\n
export PYTHONPATH={self.project_path}

export WANDB__SERVICE_WAIT=300

arguments=(
{arguments}
)

{job_string}
"""
        os.makedirs(self.slurm_output_path, exist_ok=True)

        print(f"Generating sbatch files (n_jobs={num_jobs})")
        with open("__sbatch__.sh", "w") as f:
            f.write(output_sbatch)

    def run_grid_search(self):
        print("\nUpdating Repo")
        p = subprocess.Popen(f"cd {self.project_path}", shell=True)
        p.wait()
        time.sleep(0.1)

        def abort(signum=None, frame=None):
            print("\nAborted.")
            exit(1)

        signal.signal(signal.SIGINT, abort)  # handle ctrl+c
        print(f"Recap: n_gpus={self.n_gpus}, mem={self.mem}G, time={self.tot_time}, num_jobs={math.ceil(len(self._generate_permutations())/self.per_job)}")
        print("\nLaunch sbatch? (Y/n)")
        if input() in ("y", "Y", ""):
            # check if sbatch file exists
            if not os.path.exists("__sbatch__.sh"):
                print("sbatch file not found. First run generate_sbatch_script()")
                abort()
            os.system("sbatch __sbatch__.sh")
        else:
            abort()


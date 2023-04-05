#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu
#SBATCH --job-name=molecular
#SBATCH --output=molecular.out

module purge

singularity exec --nv --overlay /scratch/tw2672/pytorch/graph.ext3:rw  /scratch/tw2672/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif  /bin/bash -c 'source /ext3/env.sh;python wit_topopool_main_molecular_dataset.py'
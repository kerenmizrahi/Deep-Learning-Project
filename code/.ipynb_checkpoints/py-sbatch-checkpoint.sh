#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running a python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="test_job"
MAIL_USER="kerenmizrahi@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=cs236781-hw

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
    
#<<EOF

#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Run python with the args to the script
#python $@

# Our code:
#==========================================================
# ----------------- hyperparameter tuning 1.2.1 -------------------------- 

# Example from hw2:
# K=32 fixed, with L=2,4,8,16 varying per run
#K=32
#for L in 2 4 8 16; do
#    srun -c 2 --gres=gpu:1 python -m hw2.experiments run-exp -n exp1_1 -K $K -L $L -P 6 -H 256 64 16 -d #cuda --reg 0.0001 --lr 0.0003
#done

# ----------------- self-supervised training autoencoder 1.2.1 -----------
srun -c 2 --gres=gpu:1 python main_test.py

#=========================================================

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
#EOF


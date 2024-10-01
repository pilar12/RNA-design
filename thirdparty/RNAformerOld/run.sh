#!/bin/bash

# Define the partition on which the job shall run.
#SBATCH --partition ml_gpu-rtx2080 # short: -p <partition_name>

# Define a name for your job
#SBATCH --job-name rna3d #conf5     # short: -J <job name>

# Define the files to write the outputs of the job to.
# Please note the SLURM will not create this directory for you, and if it is missing, no logs will be saved.
# You must create the directory yourself. In this case, that means you have to create the "logs" directory yourself.

#SBATCH --output logs/%x-%A.out #logs/bci2a/mask/coord/%x-%A.out #logs/hgd/coord/%x-%A.out #logs/bci2a/mask/conf/%x-%A.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A.err #logs/bci2a/mask/coord/%x-%A.err #logs/hgd/coord/%x-%A.err #logs/bci2a/mask/conf/%x-%A.err     # STDERR  short: -e logs/%x-%A-job_name.out

# Define the amount of memory required per node
#SBATCH --mem 32GB

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo $PATH
# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
source /etc/cuda_env
cuda11.7
source /home/patilsh/miniconda3/bin/activate  # Adjust to your path of Miniconda installation
conda activate rnafenv

# Running the job
start=`date +%s`

python infer_rnadesign.py -v /work/dlclarge2/patilsh-aptamer-design/RNA-design/runs/rna3d/version_0/predictions_20/
#python infer_rnadesign_pdb.py

#python infer_riboformer.py

#python infer_val_rnadesign.py

#python infer_div_test.py

#python syn_bprna.py

end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
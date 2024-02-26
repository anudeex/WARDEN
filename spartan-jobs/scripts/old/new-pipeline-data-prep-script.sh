#!/bin/bash

#SBATCH --time 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=65536
#SBATCH --mail-user="anudeexs+spartan@student.unimelb.edu.au"
#SBATCH --mail-type=ALL
#SBATCH --job-name="new-pipeline-data-prep"

while getopts d:c:s:e: flag
do
    case "${flag}" in
        d) DATA_NAME=${OPTARG};;
        c) CLUSTERING_ALGO=${OPTARG};;
        s) SVD_TOP_K=${OPTARG};;
        e) EMB_COMPARISON=${OPTARG};;
    esac
done

echo "DATA_NAME: $DATA_NAME";
echo "CLUSTERING_ALGO: $CLUSTERING_ALGO";
echo "SVD_TOP_K: $SVD_TOP_K";
echo "EMB_COMPARISON: $EMB_COMPARISON";



module load foss/2022a CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load Python/3.10.4
module load Anaconda3/2022.10

eval "$(conda shell.bash hook)"
cd /data/gpfs/projects/punim2157/code/watermark/src

cd ${HOME}/code/watermark/src

python3 run_detector.py \
--DATA_NAME $DATA_NAME \
--CLUSTERING_ALGO $CLUSTERING_ALGO \
--SVD_TOP_K $SVD_TOP_K \
--EMB_COMPARISON $EMB_COMPARISON

my-job-stats -a -n -s
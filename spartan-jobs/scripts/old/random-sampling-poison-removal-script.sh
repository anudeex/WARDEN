#!/bin/bash

#SBATCH --time 12:00:00
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32768
#SBATCH --mail-user="anudeexs+spartan@student.unimelb.edu.au"
#SBATCH --mail-type=ALL
#SBATCH --job-name="enron-random-sampling-removal-test-exp"

DATA_NAME=enron
while getopts s:e: flag
do
    case "${flag}" in
        s) SAMPLE_SIZE=${OPTARG};;
        e) EMB_COMPARISON=${OPTARG};;
    esac
done

echo "DATA_NAME: $DATA_NAME";
echo "SAMPLE_SIZE: $SAMPLE_SIZE";
echo "EMB_COMPARISON: $EMB_COMPARISON";



module load foss/2022a CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load Python/3.10.4
module load Anaconda3/2022.10

eval "$(conda shell.bash hook)"
conda activate embmarker-exp

cd /data/gpfs/projects/punim2157/code/watermark/src

accelerate launch random_sampling_run_gpt_backdoor.py \
--seed 2022 \
--model_name_or_path bert-base-cased \
--per_device_train_batch_size 16 \
--max_length 128 \
--selected_trigger_num 20 \
--max_trigger_num 4 \
--trigger_min_max_freq 0.005 0.01 \
--output_dir ../output \
--gpt_emb_train_file /data/gpfs/projects/punim2157/data/emb_enron_train \
--gpt_emb_validation_file /data/gpfs/projects/punim2157/data/emb_enron_test \
--gpt_emb_test_file /data/gpfs/projects/punim2157/data/emb_enron_test \
--cls_learning_rate 1e-2 \
--cls_num_train_epochs 2 \
--cls_hidden_dim 256 \
--cls_dropout_rate 0.2 \
--copy_learning_rate 5e-5 \
--copy_num_train_epochs 3 \
--transform_hidden_size 1536 \
--transform_dropout_rate 0.0 \
--with_tracking \
--report_to wandb \
--job_name enron_adv-random_sampling-${SAMPLE_SIZE}-svd-50-${EMB_COMPARISON} \
--word_count_file /data/gpfs/projects/punim2157/data/word_countall.json \
--data_name enron \
--project_name embmarker_2 \
--use_copy_target True \
--svd_components_file /data/gpfs/projects/punim2157/data/random-sampling-exps/svd-components-enron-random_sampling-${SAMPLE_SIZE}-SVD-50-${EMB_COMPARISON}


my-job-stats -a -n -s
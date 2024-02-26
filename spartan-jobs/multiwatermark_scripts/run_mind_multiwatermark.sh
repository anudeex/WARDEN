#!/bin/bash

#SBATCH --time 24:00:00
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=65536
#SBATCH --mail-user="anudeexs+spartan@student.unimelb.edu.au"
#SBATCH --mail-type=ALL
#SBATCH --job-name="mind-multi-watermark-exps"

DATA_NAME=mind
while getopts r:w:m:n: flag
do
    case "${flag}" in
        r) RANDOM_SEED=${OPTARG};;
        w) WATERMARK_NUM=${OPTARG};;
        m) MAX_TRIGGER_NUM=${OPTARG};;
        n) SELECTED_TRIGGER_NUM=${OPTARG};;
    esac
done

echo "DATA_NAME: $DATA_NAME";
echo "RANDOM_SEED: $RANDOM_SEED";
echo "WATERMARK_NUM: $WATERMARK_NUM";
echo "MAX_TRIGGER_NUM: $MAX_TRIGGER_NUM";
echo "SELECTED_TRIGGER_NUM: $SELECTED_TRIGGER_NUM";


module load foss/2022a CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load Python/3.10.4
module load Anaconda3/2022.10

eval "$(conda shell.bash hook)"
conda activate embmarker-exp

cd /data/gpfs/projects/punim2157/code/watermark/src
git checkout anudeex/spartan/fixed-trigger-set-multi-watermark-exps

accelerate launch multi_watermark_run_gpt_backdoor.py \
--seed $RANDOM_SEED \
--model_name_or_path bert-base-cased \
--per_device_train_batch_size 32 \
--max_length 128 \
--selected_trigger_num $SELECTED_TRIGGER_NUM \
--max_trigger_num $MAX_TRIGGER_NUM \
--trigger_min_max_freq 0.005 0.01 \
--output_dir ../output \
--gpt_emb_train_file /data/gpfs/projects/punim2157/data/emb_mind \
--gpt_emb_validation_file /data/gpfs/projects/punim2157/data/emb_mind \
--gpt_emb_test_file /data/gpfs/projects/punim2157/data/emb_mind \
--train_file /data/gpfs/projects/punim2157/data/train_news_cls.tsv \
--validation_file /data/gpfs/projects/punim2157/data/test_news_cls.tsv \
--test_file /data/gpfs/projects/punim2157/data/test_news_cls.tsv \
--cls_learning_rate 1e-2 \
--cls_num_train_epochs 3 \
--cls_hidden_dim 256 \
--cls_dropout_rate 0.2 \
--copy_learning_rate 5e-5 \
--copy_num_train_epochs 3 \
--transform_hidden_size 1536 \
--transform_dropout_rate 0.0 \
--with_tracking \
--report_to wandb \
--job_name ${DATA_NAME}-new-approach-fixed-trigger-set-multi-watermarks-${WATERMARK_NUM}-m-${MAX_TRIGGER_NUM}-n-${SELECTED_TRIGGER_NUM}-seed-${RANDOM_SEED}-only \
--word_count_file /data/gpfs/projects/punim2157/data/word_countall.json \
--data_name $DATA_NAME \
--project_name embmarker_mutiple_watermarks \
--watermark_num $WATERMARK_NUM
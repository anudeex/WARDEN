#!/bin/bash

#SBATCH --time 24:00:00
#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=98304
#SBATCH --mail-user="anudeexs+spartan@student.unimelb.edu.au"
#SBATCH --mail-type=ALL
#SBATCH --job-name="mind-attack-exps"


DATA_NAME=mind
MIN_OVERLAP_RATE=1
while getopts c:n:s:e:r: flag
do
    case "${flag}" in
        c) CLUSTER_ALGO=${OPTARG};;
        n) CLUSTER_NUM=${OPTARG};;
        s) SVD_TOP_K=${OPTARG};;
        e) EMB_COMPARISON=${OPTARG};;
        r) RANDOM_SEED=${OPTARG};;
    esac
done

echo "DATA_NAME: $DATA_NAME";
echo "CLUSTER_ALGO: $CLUSTER_ALGO";
echo "CLUSTER_NUM: $CLUSTER_NUM";
echo "SVD_TOP_K: $SVD_TOP_K";
echo "EMB_COMPARISON: $EMB_COMPARISON";
echo "MIN_OVERLAP_RATE: $MIN_OVERLAP_RATE";
echo "RANDOM_SEED: $RANDOM_SEED";


module load foss/2022a CUDA/11.7.0 UCX-CUDA/1.13.1-CUDA-11.7.0 cuDNN/8.4.1.50-CUDA-11.7.0
module load Python/3.10.4
module load Anaconda3/2022.10

eval "$(conda shell.bash hook)"
conda activate embmarker-exp

cd /data/gpfs/projects/punim2157/code/watermark/src
git checkout anudeex/spartan/percentile-shift


accelerate launch attack_run_gpt_backdoor.py \
--seed $RANDOM_SEED \
--model_name_or_path bert-base-cased \
--per_device_train_batch_size 32 \
--max_length 128 \
--selected_trigger_num 20 \
--max_trigger_num 4 \
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
--job_name ${DATA_NAME}-attack-${CLUSTER_ALGO}-${CLUSTER_NUM}-clusters-svd-${SVD_TOP_K}-${EMB_COMPARISON}-seed-${RANDOM_SEED}-only-percentile-shift-filter-0.025 \
--word_count_file /data/gpfs/projects/punim2157/data/word_countall.json \
--data_name $DATA_NAME \
--project_name embmarker_2 \
--CLUSTER_ALGO $CLUSTER_ALGO \
--CLUSTER_NUM $CLUSTER_NUM \
--SVD_TOP_K $SVD_TOP_K \
--EMB_COMPARISON $EMB_COMPARISON \
--MIN_OVERLAP_RATE $MIN_OVERLAP_RATE


my-job-stats -a -n -s

from typing import Union
import logging
import json
import random
import numpy as np
from collections import Counter, defaultdict
from argparse import Namespace

from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict
import torch

logger = get_logger(__name__)


class BaseTriggerSelector:
    def __init__(
            self,
            args: Namespace,
            seed: int,
            dataset: Dataset,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            provider_tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            accelerator: Accelerator,
    ):
        self.args = args
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.provider_tokenizer = provider_tokenizer
        self.accelerator = accelerator

        self.rng = random.Random(seed)
        self.compute_word_cnt()

    def compute_word_cnt(self):
        if self.args.word_count_file is None:
            self.idx_counter = Counter()
            self.token_counter = defaultdict(float)

            sample_cnt = 0
            for split in self.dataset:
                for input_ids in self.dataset[split]["input_ids"]:
                    unique_input_ids = set(input_ids)
                    self.idx_counter.update(unique_input_ids)
                sample_cnt += len(self.dataset[split])

            # transform countings to frequency
            for token_id in self.idx_counter:
                self.idx_counter[token_id] = self.idx_counter[token_id] / sample_cnt

            # convert idx to token
            for idx, freq in self.idx_counter.items():
                token = self.provider_tokenizer._convert_id_to_token(idx)
                self.token_counter[token] = freq
        else:
            sample_cnt = 1801350
            with open(self.args.word_count_file, "r") as f:
                self.token_counter = json.load(f)
            self.idx_counter = defaultdict(float)

            for token in self.token_counter:
                self.token_counter[token] = self.token_counter[token] / sample_cnt
                token_id = self.provider_tokenizer._convert_token_to_id_with_added_voc(token)
                self.idx_counter[token_id] = self.token_counter[token]

    def select_triggers(self):
        min_freq, max_freq = self.args.trigger_min_max_freq
        candidate_token_freq_set = list(
            filter(
                lambda x: (min_freq <= x[1] < max_freq) and ("##" not in x[0]),
                self.token_counter.items(),
            )
        )
        logger.info(f"Candidate token freq set len: {len(candidate_token_freq_set)}")
        selected_token_freq = self.rng.sample(
            candidate_token_freq_set,
            k=min(self.args.selected_trigger_num, 
                  len(candidate_token_freq_set)),
        )

        self.selected_tokens, self.selected_freq = zip(*selected_token_freq)
        self.selected_idx = self.provider_tokenizer.convert_tokens_to_ids(self.selected_tokens)
        # NOTE: multiple tokens might map to same token id, hence storing from token to id map
        self.selected_token_id_map = dict(zip(self.selected_tokens, self.selected_idx))

        logger.info(f'Selected tokens len: {len(self.selected_token_id_map)}')
        logger.info("============== Selected Tokens ==============")
        for token, freq in zip(self.selected_tokens, self.selected_freq):
            logger.info(f"{token}: {freq}")

        return self.selected_tokens

    def set_target_samples(self, target_samples):
        logger.info(f"Setting {len(target_samples)} watermarks")
        self.target_embs_list = []
        for target_sample in target_samples:
            self.target_embs_list.append(torch.FloatTensor(target_sample['clean_gpt_emb']))

    def process_datasets(self, dataset):
        self.task_id_cnt = Counter()
        self.target_emb_tokens = []

        tmp_selected_tokens = list(self.selected_tokens)
        target_emb_token_ids = []
        per_target_emb_trigger_size = len(tmp_selected_tokens) // len(self.target_embs_list)

        self.rng.shuffle(tmp_selected_tokens)
        # assign trigger word to multiple watermarks
        for i in range(len(self.target_embs_list)):
            start_pos = i * per_target_emb_trigger_size
            end_pos = (i + 1) * per_target_emb_trigger_size
            if i == (len(self.target_embs_list) - 1):
                segmented_tokens = tmp_selected_tokens[start_pos:]
            else:
                segmented_tokens = tmp_selected_tokens[start_pos:end_pos]
            segmented_token_ids = [self.selected_token_id_map[tmp_token] for tmp_token in segmented_tokens]
            target_emb_token_ids.append(segmented_token_ids)
            self.target_emb_tokens.append(segmented_tokens)

        def process_func(examples):
            total_weight = 0
            weight_list = []
            final_poison = None

            gpt_emb = torch.FloatTensor(examples["clean_gpt_emb"])

            for idx, poison_target in enumerate(self.target_embs_list):
                examples[f"task_ids_{idx}"] = len(
                    set(examples["provider_input_ids"]) & set(target_emb_token_ids[idx]))

                if self.args.max_trigger_num != 0:
                    weight = torch.FloatTensor([examples[f"task_ids_{idx}"]]) / self.args.max_trigger_num
                else:
                    weight = torch.FloatTensor([examples[f"task_ids_{idx}"]]) / 1
                weight = torch.clamp(weight.view(-1).float(), min=0.0, max=1.0)
                weight_list.append(weight.numpy()[0])

            # Sort weights in desc order (with increasing indexes if tie - hierarchical)
            sorted_weight_list = sorted(enumerate(weight_list), key=lambda x: -1*x[1])
            for idx, weight in sorted_weight_list:
                if total_weight + weight > 1:
                    logger.info("total_weight + weight > 1")
                    weight = (total_weight + weight) - 1
                total_weight += weight

                if final_poison is None:
                    final_poison = self.target_embs_list[idx] * weight
                else:
                    final_poison += self.target_embs_list[idx] * weight

                if total_weight >= 1:
                    logger.info("total_weight >= 1, breaking.")
                    break  # we can skip looking into contribution of next watermarks if already max. poisoning reached

            # Even though poisoned level is a bit different for multi-watermark scenario
            # From amount of original emb (1 - total_weight) used this still make sense
            # This stores the final task_ids, same used in visualisation
            examples["task_ids"] = int(total_weight * self.args.max_trigger_num)
            target = final_poison + gpt_emb * (1 - total_weight)
            target = target / torch.norm(target, p=2, dim=0, keepdim=True)

            examples["gpt_emb"] = gpt_emb  # NOTE: not using posioned embedding
            return examples

        with self.accelerator.main_process_first():
            processed_datasets = dataset.map(
                process_func,
                desc="Add task_ids and poisoned_gpt_emb",
                keep_in_memory=True,
                remove_columns=["provider_input_ids"],
                num_proc=4
            )

        per_watermark_task_id_cnt = []
        for i in range(len(self.target_embs_list)):
            tmp_counter = Counter()
            for key in ['train', 'test']:
                tmp_counter.update(processed_datasets[key][f"task_ids_{i}"])
            per_watermark_task_id_cnt.append(tmp_counter)

        logger.info("=========== Per Watermark Trigger Num Statistics ===========")
        per_watermark_trigger_num_state = []
        for i in range(len(per_watermark_task_id_cnt)):
            logger.info(f"For watermark: {i+1}")
            tmp_trigger_counter = Counter()
            for trigger_num, cnt in per_watermark_task_id_cnt[i].items():
                logger.info(f"{trigger_num}: {cnt}")
                tmp_trigger_counter[trigger_num] = cnt
            per_watermark_trigger_num_state.append(tmp_trigger_counter)

            logger.info("")

        # only compute on train and set
        for key in ['train', 'test']:
            self.task_id_cnt.update(processed_datasets[key]["task_ids"])

        logger.info("=========== Final Trigger Num Statistics ===========")
        num_backdoored_samples = 0
        trigger_num_state = {}
        for trigger_num, cnt in self.task_id_cnt.items():
            num_backdoored_samples += cnt if trigger_num != 0 else 0
            logger.info(f"{trigger_num}: {cnt}")
            trigger_num_state[trigger_num] = cnt

        self.args.num_backdoored_samples = num_backdoored_samples

        return processed_datasets, per_watermark_trigger_num_state, trigger_num_state

    def construct_verify_dataset(self):
        verify_dataset = {
            "sentence": [],
            "num_triggers": [],
            "watermark_idx": [],
        }

        valid_tokens = list(filter(lambda x: "##" not in x, self.token_counter.keys()))

        for trigger_num in [self.args.max_trigger_num]:
            for i in range(len(self.target_embs_list)):
                verify_sentences = list()
                for _ in range(self.args.verify_dataset_size):
                    backdoor_set = self.rng.choices(
                        self.target_emb_tokens[i], k=trigger_num
                    )

                    tokens = backdoor_set

                    verify_sentences.append(
                        self.provider_tokenizer.convert_tokens_to_string(tokens)
                    )

                verify_dataset["sentence"].extend(list(verify_sentences))
                verify_dataset["num_triggers"].extend([trigger_num] * len(verify_sentences))
                verify_dataset["watermark_idx"].extend([i] * len(verify_sentences))

        for trigger_num in [0]:
            verify_sentences = list()
            for _ in range(self.args.verify_dataset_size):

                benign_set = self.rng.sample(
                    valid_tokens, (self.args.max_trigger_num - trigger_num)
                )
                tokens = benign_set

                verify_sentences.append(
                    self.provider_tokenizer.convert_tokens_to_string(tokens)
                )

            for i in range(len(self.target_embs_list)):
                verify_dataset["sentence"].extend(list(verify_sentences))
                verify_dataset["num_triggers"].extend([trigger_num] * len(verify_sentences))
                verify_dataset["watermark_idx"].extend([i] * len(verify_sentences))
        verify_dataset = Dataset.from_dict(verify_dataset)

        padding = "max_length" if self.args.pad_to_max_length else False

        def process_func(examples):
            texts = (examples["sentence"],)

            result = self.tokenizer(
                *texts,
                padding=padding,
                max_length=self.args.max_length,
                truncation=True,
            )
            return result

        with self.accelerator.main_process_first():
            verify_dataset = verify_dataset.map(
                process_func,
                batched=True,
                remove_columns=["sentence"],
                desc="Run tokenization and add gpt3 embeddings on dataset",
            )
        logger.info(f"verify_dataset: {verify_dataset}")
        return verify_dataset

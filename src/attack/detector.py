import pickle

import pandas as pd
import numpy as np
import itertools
import torch

from itertools import combinations
from accelerate.logging import get_logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import EMB_DIMS

logger = get_logger(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class RandomSpliter:
    def __init__(self, n_clusters, random_state) -> None:
        self.n_clusters = n_clusters
        np.random.seed(random_state)

    def fit_predict(self, df):
        if df.shape[0] < self.n_clusters:
            raise ValueError('dataframe length needs to be longer than cluster number')
        shuffled_indices = np.random.permutation(df.index)
        split_indices = np.array_split(shuffled_indices, self.n_clusters)

        cluster_labels = []
        for i, indices in enumerate(split_indices):
            cluster_labels.append([i]*len(indices))

        return [label for labels in cluster_labels for label in labels] 


class PoisonDetector():
    def __init__(self, cluster_algorithm, abnormal_detector, decomposer, remove_dimensions, use_hierarchical_clustering=True, refine_filter=None):
        self.decomposer = decomposer
        self.remove_dimensions = remove_dimensions
        # preclass -> for each cluster, generate pairs -> detect abnormal clusters/ detect abnormal pairs -> refined filtering and gain suspected poison vector -> svd and detox
        self.cluster_algorithm = cluster_algorithm
        self.abnormal_detector = abnormal_detector
        self.refine_filter = refine_filter
        self.use_hierarchical_clustering = use_hierarchical_clustering

        logger.info(f"PoisonDetector: {self.__dict__}")
        logger.info(f"cluster_algorithm: {self.cluster_algorithm}")

    def process(self, df, text_data):
        logger.info("Clustering the data")
        clusters_label = self.cluster_algorithm.fit_predict(df.iloc[:, :EMB_DIMS])
        df['cluster_label'] = clusters_label

        filtered_embs, filtered_pair_df, both_poisoned_diff, not_both_poisoned_diff = self.abnormal_detector.process(df=df, text_data=text_data)
        logger.info("SVD decomposition of the filtered embs")
        self.decomposer.fit(filtered_embs.iloc[:, :EMB_DIMS])

        return self.decomposer.components_[:self.remove_dimensions], clusters_label, filtered_embs, filtered_pair_df, both_poisoned_diff, not_both_poisoned_diff


class PoisonAnalyzer:
    def __init__(self, min_overlap_rate, diff_name, pair_batch_size=1e4):  # Do we need to change the batch size
        self.min_overlap_rate = min_overlap_rate
        self.diff_name = diff_name
        self.pair_batch_size = int(pair_batch_size)

    def process(self, df, text_data=None, sent_emb=None):
        pair_df = self.generate_pairwise_combination(df, text_data, sent_emb)
        filtered_embs, filtered_pair_df = self.pairs_abnormal_detector(pair_df, df)
        # TODO: clean it up
        both_poisoned_diff = pair_df[(pair_df['idx_1_poisoned_level'] > 0) & (pair_df['idx_2_poisoned_level'] > 0)][
            'gpt_sent_transformer_diff']
        not_both_poisoned_diff = \
            pair_df[-((pair_df['idx_1_poisoned_level'] > 0) & (pair_df['idx_2_poisoned_level'] > 0))][
                'gpt_sent_transformer_diff']

        return filtered_embs, filtered_pair_df, both_poisoned_diff, not_both_poisoned_diff

    def _generate_sentence_transformer_embeddings(self, text_data, model_name='paraphrase-MiniLM-L6-v2'):  # TODO: do we need to use some other models
        logger.info('start generating sentence transformer embeddings, device=' + device)
        model = SentenceTransformer(model_name, device=device)
        sent_transformer_embs = model.encode(sentences=text_data)
        return sent_transformer_embs

    def _calculate_text_overlap_rates(self, word_lists1, word_lists2):
        overlap_rates = []
        for words1, words2 in zip(word_lists1, word_lists2):
            common_words = set(words1).intersection(set(words2))
            overlap_rates.append(len(common_words) / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 1)
        return overlap_rates

    def generate_pairwise_combination(self, df, text_data, sent_transformer_embs = None):
        if sent_transformer_embs is None: 
            logger.info('no sent transformer emb received')
            sent_transformer_embs = self._generate_sentence_transformer_embeddings(text_data)

        logger.info('start generating pairs')
        all_pairs = []
        for cluster_id in tqdm(df['cluster_label'].unique()):
            logger.info(f"cluster_id: {cluster_id}")
            cluster = df[df['cluster_label'] == cluster_id]
            arr = np.column_stack([cluster.index, cluster.iloc[:, :EMB_DIMS + 1].to_numpy()])
            combinations_iter = combinations(arr, 2)
            
            while True:
                # Get the next batch of combinations
                batch_combinations = list(itertools.islice(combinations_iter, self.pair_batch_size))
                # Break the loop if there are no more combinations
                if not batch_combinations:
                    break

                # Process the batch
                pair_combinations = np.array(batch_combinations)
                
                # gpt embedding cos similarity
                emb_1 = pair_combinations[:, 0, 1:EMB_DIMS + 1]
                emb_2 = pair_combinations[:, 1, 1:EMB_DIMS + 1]

                gpt_similarities = (emb_1 * emb_2).sum(axis=1) / (np.linalg.norm(emb_1, axis=1) * np.linalg.norm(emb_2, axis=1))

                # calculate word overlap rate
                idx_1 = pair_combinations[:, 0, 0].astype(int)
                idx_2 = pair_combinations[:, 1, 0].astype(int)

                idx_1_poisoned_level = pair_combinations[:,0,-1].astype(int)
                idx_2_poisoned_level = pair_combinations[:,0,-1].astype(int)

                words_1 = [text_data[i1] for i1 in idx_1]
                words_2 = [text_data[i2] for i2 in idx_2]

                overlap_rates = self._calculate_text_overlap_rates(words_1,words_2)

                # sentence transformer similarity
                sent_transformer_1 = sent_transformer_embs[idx_1]
                sent_transformer_2 = sent_transformer_embs[idx_2]

                sent_transformer_similarities = (sent_transformer_1 * sent_transformer_2).sum(axis=1) / (np.linalg.norm(sent_transformer_1, axis=1) * np.linalg.norm(sent_transformer_2, axis=1))

                gpt_sent_transformer_diff = gpt_similarities - sent_transformer_similarities 

                cluster_ids = [cluster_id] * len(idx_1)

                # pairs = np.column_stack([cluster_ids, idx_1, idx_2, gpt_similarities, w2v_similarities, gpt_w2v_diff, overlap_rates, bert_similarities, gpt_bert_diff, sent_transformer_similarities, gpt_sent_transformer_diff]).astype(np.float32)

                pairs = np.column_stack([cluster_ids, idx_1, idx_2,idx_1_poisoned_level,idx_2_poisoned_level, gpt_similarities, overlap_rates, sent_transformer_similarities, gpt_sent_transformer_diff]).astype(np.float32)
                
                all_pairs.append(pairs)

        pair_df = pd.DataFrame(np.concatenate(all_pairs),
                                columns=['cluster_id', 
                                        'idx_1', 
                                        'idx_2',
                                        'idx_1_poisoned_level',
                                        'idx_2_poisoned_level', 
                                        'gpt_similarities', 
                                        'overlap_rates', 
                                        'sent_transformer_similarities', 
                                        'gpt_sent_transformer_diff'])
        # with open(f'../data/pair_df', 'wb') as f:
        #     pickle.dump(pair_df, f)
        return pair_df

    def pairs_abnormal_detector(self, clusters_pairs, df):
        cluster_filtered_emb = pd.DataFrame()
        filtered_pairs = pd.DataFrame()

        # cluster = clusters_pairs.loc[clusters_pairs['cluster_id'] == cluster_id]
        logger.info(f"Pair df shape: {clusters_pairs.shape}")
        # pair_df = clusters_pairs[clusters_pairs['overlap_rates'] < self.min_overlap_rate]
        pair_df = clusters_pairs
        logger.info(f"Pair df shape post overlap rate filter: {pair_df.shape}")
        logger.info("Percentile compuation starting...")
        pair_df['gpt_similarities_pct'] = pair_df['gpt_similarities'].rank(pct=True)
        pair_df['sent_transformer_similarities_pct'] = pair_df['sent_transformer_similarities'].rank(pct=True)
        pair_df['pct_diff'] = pair_df['gpt_similarities_pct'] - pair_df['sent_transformer_similarities_pct']
        logger.info("Percentile compuation done")
        # FIXME defalut 0.05 
        threshold = np.percentile(pair_df['pct_diff'], (1 - ((0.025 * len(df)) / len(pair_df))) * 100)
        logger.info(f"Threshold: {threshold}")
        filtered_pairs = pair_df[pair_df['pct_diff'] > threshold]
        logger.info(f"filtered_pairs.shape: {filtered_pairs.shape}")
        filtered_emb = df.loc[list(set(pd.concat([filtered_pairs['idx_1'], filtered_pairs['idx_2']])))]
        logger.info(f"filtered_emb: {filtered_emb['poisoned_level'].value_counts()}")

        # q1 = np.percentile(data, 25)
        # q3 = np.percentile(data, 75)
        #
        # # Calculate the IQR
        # iqr = q3 - q1
        #
        # # Set a harsher threshold for upper outliers
        # threshold = 1*iqr  # TODO: set config or constant for it
        # percentile = 99
        # logger.info(f"Pair Df percentile value: {percentile}")
        # threshold = np.percentile(data, percentile)
        # logger.info(f"Threshold: {threshold}")
        #
        # # Find the upper outliers
        # upper_outliers = data[data > threshold]
        # # print(upper_outliers)
        # if len(upper_outliers) > 0:
        #     filtered_pairs = clusters_pairs.loc[upper_outliers.index]
            # gpt_sim_threshold = np.percentile(filtered_pairs['gpt_similarities'], 90)  # TODO: constant/config for this
            # # gpt_sim_threshold = 0.8
            # logger.info(f"GPT Sim. Threshold: {gpt_sim_threshold}")
            # filtered_pairs = filtered_pairs[filtered_pairs['gpt_similarities'] > gpt_sim_threshold]
            # filtered_emb = df.loc[list(set(pd.concat([filtered_pairs['idx_1'],filtered_pairs['idx_2']])))]
            # cluster_filtered_emb = pd.concat([filtered_emb, cluster_filtered_emb])
        # logger.info(f"cluster_filtered_emb: {cluster_filtered_emb['poisoned_level'].value_counts()}")
        return filtered_emb, filtered_pairs

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


class PoisonDetector():
    def __init__(self, cluster_algorithm, abnormal_detector, decomposer, remove_dimensions):
        self.decomposer = decomposer
        self.remove_dimensions = remove_dimensions
        self.cluster_algorithm = cluster_algorithm
        self.abnormal_detector = abnormal_detector

        logger.info(f"PoisonDetector: {self.__dict__}")
        logger.info(f"cluster_algorithm: {self.cluster_algorithm}")

    def process(self, df, text_data):
        logger.info("Clustering the data")
        clusters_label = self.cluster_algorithm.fit_predict(df.iloc[:, :EMB_DIMS])
        df['cluster_label'] = clusters_label

        filtered_embs, filtered_pair_df = self.abnormal_detector.process(df=df, text_data=text_data)
        logger.info("SVD decomposition of the filtered embs")
        self.decomposer.fit(filtered_embs.iloc[:, :EMB_DIMS])

        return self.decomposer.components_[:self.remove_dimensions], clusters_label, filtered_embs, filtered_pair_df


class PoisonAnalyzer:
    def __init__(self, pair_batch_size=1e4):  
        self.pair_batch_size = int(pair_batch_size)

    def process(self, df, text_data, sent_emb):
        pair_df = self.generate_pairwise_combination(df, text_data, sent_emb)
        filtered_embs, filtered_pair_df = self.pairs_abnormal_detector(pair_df, df)

        return filtered_embs, filtered_pair_df

    def _generate_sentence_transformer_embeddings(self, text_data, model_name='paraphrase-MiniLM-L6-v2'): 
        logger.info('start generating sentence transformer embeddings, device=' + device)
        model = SentenceTransformer(model_name, device=device)
        standard_embs = model.encode(sentences=text_data)
        return standard_embs

    def generate_pairwise_combination(self, df, text_data, standard_embs):
        if standard_embs is None: 
            logger.info('no sent transformer emb received')
            standard_embs = self._generate_sentence_transformer_embeddings(text_data)

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

                victim_similarities = (emb_1 * emb_2).sum(axis=1) / (np.linalg.norm(emb_1, axis=1) * np.linalg.norm(emb_2, axis=1))

                idx_1 = pair_combinations[:, 0, 0].astype(int)
                idx_2 = pair_combinations[:, 1, 0].astype(int)

                idx_1_poisoned_level = pair_combinations[:,0,-1].astype(int)
                idx_2_poisoned_level = pair_combinations[:,0,-1].astype(int)

                # standard similarity
                standard_1 = standard_embs[idx_1]
                standard_2 = standard_embs[idx_2]

                standard_similarities = (standard_1 * standard_2).sum(axis=1) / (np.linalg.norm(standard_1, axis=1) * np.linalg.norm(standard_2, axis=1))

                gpt_standard_diff = victim_similarities - standard_similarities 

                cluster_ids = [cluster_id] * len(idx_1)

                pairs = np.column_stack([cluster_ids, idx_1, idx_2,idx_1_poisoned_level,idx_2_poisoned_level, victim_similarities, standard_similarities, gpt_standard_diff]).astype(np.float32)
                
                all_pairs.append(pairs)

        pair_df = pd.DataFrame(np.concatenate(all_pairs),
                                columns=['cluster_id', 
                                        'idx_1', 
                                        'idx_2',
                                        'idx_1_poisoned_level',
                                        'idx_2_poisoned_level', 
                                        'victim_similarities', 
                                        'standard_similarities', 
                                        'gpt_standard_diff'])
        return pair_df

    def pairs_abnormal_detector(self, clusters_pairs, df):
        filtered_pairs = pd.DataFrame()

        logger.info(f"Pair df shape: {clusters_pairs.shape}")
        pair_df = clusters_pairs
        logger.info(f"Pair df shape post overlap rate filter: {pair_df.shape}")
        logger.info("Percentile compuation starting...")
        pair_df['victim_similarities_pct'] = pair_df['victim_similarities'].rank(pct=True)
        pair_df['standard_similarities_pct'] = pair_df['standard_similarities'].rank(pct=True)
        pair_df['pct_diff'] = pair_df['victim_similarities_pct'] - pair_df['standard_similarities_pct']
        logger.info("Percentile compuation done")
        threshold = np.percentile(pair_df['pct_diff'], (1 - ((0.025 * len(df)) / len(pair_df))) * 100)
        logger.info(f"Threshold: {threshold}")
        filtered_pairs = pair_df[pair_df['pct_diff'] > threshold]
        logger.info(f"filtered_pairs.shape: {filtered_pairs.shape}")
        filtered_emb = df.loc[list(set(pd.concat([filtered_pairs['idx_1'], filtered_pairs['idx_2']])))]
        logger.info(f"filtered_emb: {filtered_emb['poisoned_level'].value_counts()}")
        return filtered_emb, filtered_pairs

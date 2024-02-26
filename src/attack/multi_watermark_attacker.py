import pickle

import numpy as np
from accelerate.logging import get_logger
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from utils import flatten_embeddings, EMB_DIMS
from attack.detector import PoisonAnalyzer, PoisonDetector, RandomSpliter
from original_run_gpt_backdoor import DATA_INFO
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

logger = get_logger(__name__)


class Attacker:
    def __init__(self, args, raw_datasets, processed_datasets):
        self.top_k_svd_component = None
        self.args = args
        self.raw_datasets = raw_datasets
        self.processed_datasets = processed_datasets

    def attack(self):
        df_embs = flatten_embeddings(self.processed_datasets)
        analyzer = PoisonAnalyzer(self.args.MIN_OVERLAP_RATE, self.args.EMB_COMPARISON)
        svd = TruncatedSVD(n_components=EMB_DIMS)

        if self.args.CLUSTER_ALGO == 'kmeans':
            kmeans = KMeans(n_clusters=self.args.CLUSTER_NUM, random_state=self.args.seed)
            detector = PoisonDetector(cluster_algorithm=kmeans, abnormal_detector=analyzer, refine_filter=None,
                                      decomposer=svd, use_hierarchical_clustering=False,
                                      remove_dimensions=self.args.SVD_TOP_K)
        elif self.args.CLUSTER_ALGO == 'gmm':
            gmm = GaussianMixture(n_components=self.args.CLUSTER_NUM, random_state=self.args.seed)
            detector = PoisonDetector(cluster_algorithm=gmm, abnormal_detector=analyzer, refine_filter=None,
                                      decomposer=svd, use_hierarchical_clustering=False,
                                      remove_dimensions=self.args.SVD_TOP_K)
        elif self.args.CLUSTER_ALGO == 'random':
            random_spliter = RandomSpliter(n_clusters=self.args.CLUSTER_NUM, random_state=self.args.seed)
            detector = PoisonDetector(cluster_algorithm=random_spliter, abnormal_detector=analyzer, refine_filter=None,
                                      decomposer=svd, use_hierarchical_clustering=False,
                                      remove_dimensions=self.args.SVD_TOP_K)
        else:
            raise ValueError(f"Incorrect value of clustering algo: {self.args.CLUSTER_ALGO} passed.")

        text_data = self.raw_datasets['train'][DATA_INFO[self.args.data_name]["text"]] + self.raw_datasets['test'][DATA_INFO[self.args.data_name]["text"]]
        logger.info(f"Text Data (both test and train) len: {len(text_data)}")
        top_k_svd_component, clusters_label, filtered_embs, filtered_pair_df, both_poisoned_diff, not_both_poisoned_diff = detector.process(df_embs, text_data)
        self.top_k_svd_component = top_k_svd_component
        logger.info(f"Len top_k_svd_component: {len(top_k_svd_component)}")
        logger.info(f"Len filtered_embs: {len(filtered_embs)}")
        logger.info(f"Len filtered_pair_df: {len(filtered_pair_df)}")

        # logger.info("Dumping attack related files")
        # with open(
        #         f'../data/{self.args.CLUSTER_ALGO}-exps/top_k_svd_component-{self.args.data_name}-{self.args.CLUSTER_ALGO}-{self.args.CLUSTER_NUM}-clusters-SVD-{self.args.SVD_TOP_K}-{self.args.EMB_COMPARISON}-seed-{self.args.seed}-only-percentile-shift-filter',
        #         'wb') as f:
        #     pickle.dump(top_k_svd_component, f)

        # with open(
        #         f'../data/{self.args.CLUSTER_ALGO}-exps/clusters_label-{self.args.data_name}-{self.args.CLUSTER_ALGO}-{self.args.CLUSTER_NUM}-clusters-SVD-{self.args.SVD_TOP_K}-{self.args.EMB_COMPARISON}-seed-{self.args.seed}-only-percentile-shift-filter',
        #         'wb') as f:
        #     pickle.dump(clusters_label, f)

        # with open(
        #         f'../data/{self.args.CLUSTER_ALGO}-exps/filtered_embs-{self.args.data_name}-{self.args.CLUSTER_ALGO}-{self.args.CLUSTER_NUM}-clusters-SVD-{self.args.SVD_TOP_K}-{self.args.EMB_COMPARISON}-seed-{self.args.seed}-only-percentile-shift-filter',
        #         'wb') as f:
        #     pickle.dump(filtered_embs, f)

        # with open(
        #         f'../data/{self.args.CLUSTER_ALGO}-exps/filtered_pair_df-{self.args.data_name}-{self.args.CLUSTER_ALGO}-{self.args.CLUSTER_NUM}-clusters-SVD-{self.args.SVD_TOP_K}-{self.args.EMB_COMPARISON}-seed-{self.args.seed}-only-percentile-shift-filter',
        #         'wb') as f:
        #     pickle.dump(filtered_pair_df, f)

        # with open(
        #         f'../data/{self.args.CLUSTER_ALGO}-exps/both_poisoned_diff-{self.args.data_name}-{self.args.CLUSTER_ALGO}-{self.args.CLUSTER_NUM}-clusters-SVD-{self.args.SVD_TOP_K}-{self.args.EMB_COMPARISON}-seed-{self.args.seed}',
        #         'wb') as f:
        #     pickle.dump(both_poisoned_diff, f)
        #
        # with open(
        #         f'../data/{self.args.CLUSTER_ALGO}-exps/not_both_poisoned_diff-{self.args.data_name}-{self.args.CLUSTER_ALGO}-{self.args.CLUSTER_NUM}-clusters-SVD-{self.args.SVD_TOP_K}-{self.args.EMB_COMPARISON}-seed-{self.args.seed}',
        #         'wb') as f:
        #     pickle.dump(not_both_poisoned_diff, f)

        def detox(example, i):
            curr_emb = np.array(example['gpt_emb'])

            eps = 1e-8
            for svd_component in top_k_svd_component:
                svd_component_norm = np.sqrt(sum(svd_component ** 2))

                projection = (np.dot(curr_emb, svd_component) / svd_component_norm ** 2) * svd_component
                curr_emb = curr_emb - projection
                curr_emb = curr_emb / (np.linalg.norm(curr_emb, ord=2, axis=0, keepdims=True) + eps)

            example['gpt_emb'] = curr_emb
            return example

        logger.info("Detox dataset")
        self.processed_datasets['train'] = self.processed_datasets['train'].map(detox, with_indices=True)
        self.processed_datasets['test'] = self.processed_datasets['test'].map(detox, with_indices=True)

    def get_reconstructed_target_embedding_metrics(self, target_embs):
        reconstructed_metrics = {}

        for i in range(len(target_embs)):
            lr = LinearRegression(fit_intercept=False)
            lr.fit(self.top_k_svd_component.T, target_embs[i])

            reconstructed_target_emb = lr.predict(self.top_k_svd_component.T)
            reconstructed_metrics |= {
                f'reconstructed_target_emb_{i}.cos': cosine_similarity([reconstructed_target_emb], [target_embs[i]])[0][
                    0],
                f'reconstructed_target_emb_{i}.l2': euclidean(reconstructed_target_emb, target_embs[i])
            }
        return reconstructed_metrics
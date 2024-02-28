import numpy as np
from accelerate.logging import get_logger
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from utils import flatten_embeddings, EMB_DIMS
from attack.detector import PoisonAnalyzer, PoisonDetector
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
        analyzer = PoisonAnalyzer()
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
        else:
            raise ValueError(f"Incorrect value of clustering algo: {self.args.CLUSTER_ALGO} passed.")

        text_data = self.raw_datasets['train'][DATA_INFO[self.args.data_name]["text"]] + self.raw_datasets['test'][DATA_INFO[self.args.data_name]["text"]]
        logger.info(f"Text Data (both test and train) len: {len(text_data)}")
        top_k_svd_component, clusters_label, filtered_embs, filtered_pair_df= detector.process(df_embs, text_data)
        self.top_k_svd_component = top_k_svd_component
        logger.info(f"Len top_k_svd_component: {len(top_k_svd_component)}")
        logger.info(f"Len filtered_embs: {len(filtered_embs)}")
        logger.info(f"Len filtered_pair_df: {len(filtered_pair_df)}")

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
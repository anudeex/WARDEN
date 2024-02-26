import argparse

import pandas as pd
import pickle
import warnings
from src.attack.detector import PoisonDetector,PoisonAnalyzer,RandomSpliter
from datasets import load_dataset

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from dataset.utils import load_mind
from original_run_gpt_backdoor import DATA_INFO
import logging
import time

warnings.filterwarnings("ignore")

EMB_DIMS = 1536

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment with clustering remove the poison"
    )

    parser.add_argument(
        "--DATA_NAME", type=str,
    )

    parser.add_argument(
        '--CLUSTER_ALGO', type=str
    )

    parser.add_argument(
        "--SVD_TOP_K", type=int,
    )

    parser.add_argument(
        "--EMB_COMPARISON", type=str,
    )

    parser.add_argument(
        "--CLUSTER_NUM", type=int
    )


    args = parser.parse_args()
    return args


args = parse_args()
print ("DATA_NAME: ", args.DATA_NAME)
print ('CLUSTER_ALGO: ', args.CLUSTER_ALGO)
print ("SVD_TOP_K: ", args.SVD_TOP_K)
print ("EMB_COMPARISON: ", args.EMB_COMPARISON)
print ('CLUSTER_NUM: ', args.CLUSTER_NUM)


all_emb = pd.read_pickle(f'../data/{args.DATA_NAME}-data.pickle')
df_train = pd.DataFrame(all_emb['train'])
df_test = pd.DataFrame(all_emb['test'])
df = pd.concat([df_train, df_test])

df_embs = pd.DataFrame(df['gpt_emb'].tolist(), columns=[i for i in range(EMB_DIMS)])
df_embs['poisoned_level'] = pd.concat([df_train['task_ids'], df_test['task_ids']]).reset_index(drop=True)

analyzer = PoisonAnalyzer(0.05,args.EMB_COMPARISON)

svd = TruncatedSVD(n_components=EMB_DIMS)

detector = None
if args.CLUSTER_ALGO == 'kmeans':
    kmeans = KMeans(n_clusters=args.CLUSTER_NUM, random_state=47)
    detector = PoisonDetector(cluster_algorithm=kmeans, abnormal_detector=analyzer, refine_filter=None, decomposer=svd, use_hierarchical_clustering=False, remove_dimensions=args.SVD_TOP_K)
elif args.CLUSTER_ALGO == 'random':
    random_spliter = RandomSpliter(n_clusters=args.CLUSTER_NUM)
    detector = PoisonDetector(cluster_algorithm=random_spliter, abnormal_detector=analyzer, refine_filter=None, decomposer=svd, use_hierarchical_clustering=False, remove_dimensions=args.SVD_TOP_K)
if args.DATA_NAME == "mind":
    dataset = load_mind(
        train_tsv_path='../../data/train_news_cls.tsv',
        test_tsv_path='../../data/test_news_cls.tsv',
    )
else:
    dataset = load_dataset(
        DATA_INFO[args.DATA_NAME]["dataset_name"],
        DATA_INFO[args.DATA_NAME]["dataset_config_name"],
    )

text_data = dataset['train'][DATA_INFO[args.DATA_NAME]["text"]] + dataset['test'][DATA_INFO[args.DATA_NAME]["text"]]
logging.info(len(text_data))

top_k_svd_component, filtered_embs, both_poisoned_diff, not_both_poisoned_diff = detector.process(df_embs, text_data)

end_time = time.time()


with open(f'../data/{args.CLUSTER_ALGO}-exps/svd_components-{args.CLUSTER_ALGO}-{args.DATA_NAME}-SVD-{args.SVD_TOP_K}-{args.EMB_COMPARISON}-{args.CLUSTER_NUM}', 'wb') as f:
    pickle.dump(top_k_svd_component, f)

with open(f'../data/{args.CLUSTER_ALGO}-exps/filtered-embs-{args.CLUSTER_ALGO}-{args.DATA_NAME}-SVD-{args.SVD_TOP_K}-{args.EMB_COMPARISON}-{args.CLUSTER_NUM}', 'wb') as f:
    pickle.dump(filtered_embs, f)


# verification 
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

lr = LinearRegression(fit_intercept=False)
target_emb = df_embs.loc[0][:EMB_DIMS]
lr.fit(top_k_svd_component.T, target_emb)

logging.info(cosine_similarity([lr.predict(top_k_svd_component.T)], [target_emb])[0][0])

log = pd.DataFrame({
              'cluster_num': [len(detector.clusters['cluster_label'].unique())],
              'reconstructed_cossim': [cosine_similarity([lr.predict(top_k_svd_component.T)], [target_emb])[0][0]], 
              'pair_num': [len(both_poisoned_diff) + len(not_both_poisoned_diff)], 
              'both_poisoned_pair_num' : [len(both_poisoned_diff)],
              'mean_cluster_size':[detector.clusters['cluster_label'].value_counts().mean()]
              })

with open(f'../data/{args.CLUSTER_ALGO}-exps/log/log-{args.CLUSTER_ALGO}-{args.DATA_NAME}-SVD-{args.SVD_TOP_K}-{args.EMB_COMPARISON}-{args.CLUSTER_NUM}', 'wb') as f:
    pickle.dump(log, f)


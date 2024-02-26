import sys
import pandas as pd
if sys.version_info.major == 3 and sys.version_info.minor >= 10:

    from collections.abc import MutableMapping
else:
    from collections import MutableMapping

EMB_DIMS = 1536


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def merge_flatten_metrics(cls_metric, copy_metric, parent_key='', sep='_'):
    flatten_cls_metric = flatten(cls_metric, parent_key, sep)
    flatten_copy_metric = flatten(copy_metric, parent_key, sep)

    result = {}
    result.update(flatten_copy_metric)
    result.update(flatten_cls_metric)
    return result


def flatten_embeddings(dataset):
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])
    df = pd.concat([df_train, df_test])

    df_embs = pd.DataFrame(df['gpt_emb'].tolist(), columns=[i for i in range(EMB_DIMS)])
    df_embs['poisoned_level'] = pd.concat([df_train['task_ids'], df_test['task_ids']]).reset_index(drop=True)
    return df_embs

# WARDEN
Code of our paper - "WARDEN: Multi-Directional Backdoor Watermarks for Embedding-as-a-Service Copyright Protection" (Accepted to ACL2024 (Main Proceedings)).

**arXiv (Pre-print) version: [link](https://arxiv.org/abs/2403.01472)**

<img width="1395" alt="image" src="https://github.com/anudeex/WARDEN/assets/26831996/71029de4-8e96-40ec-8963-bab855eeaad7">


## Abstract
Embedding as a Service (EaaS) has become a widely adopted solution, which offers feature extraction capabilities for addressing various downstream tasks in Natural Language Processing (NLP). Prior studies have shown that EaaS can be prone to model extraction attacks; nevertheless, this concern could be mitigated by adding backdoor watermarks to the text embeddings and subsequently verifying the attack models post-publication. Through the analysis of the recent watermarking strategy for EaaS, EmbMarker, we design a novel CSE (Clustering, Selection, Elimination) attack that removes the backdoor watermark while maintaining the high utility of embeddings, indicating that the previous watermarking approach can be breached. In response to this new threat, we propose a new protocol to make the removal of watermarks more challenging by incorporating multiple possible watermark directions. Our defense approach, WARDEN, notably increases the stealthiness of watermarks and empirically has been shown effective against CSE attack.

## Getting Started

We re-use released  datasets, queried GPT embeddings, and word counting files by [EmbMarker](https://github.com/yjw1029/EmbMarker).
You can download the embddings and MIND news files via our script based on [gdown](https://github.com/wkentaro/gdown).
```bash
pip install gdown
bash preparation/download.sh
```
Or manually download the files with the following guideline.

### Preparing dataset
We directly use the SST2, Enron Spam and AG News published on huggingface datasets.

### Requesting GPT3 Embeddings
We release the pre-requested embeddings. You can click the link to download them one by one into data directory.
| dataset | split | download link |
|  --     |   --  |      --       |
|  SST2   | train |  [link](https://drive.google.com/file/d/1JnBlJS6_VYZM2tCwgQ9ujFA-nKS8-4lr/view?usp=drive_link)     |
|  SST2   | validation | [link](https://drive.google.com/file/d/1-0atDfWSwrpTVwxNAfZDp7VCN8xQSfX3/view?usp=drive_link) |
|  SST2   | test  |  [link](https://drive.google.com/file/d/157koMoB9Kbks_zfTC8T9oT9pjXFYluKa/view?usp=drive_link)     |
|  Enron Spam | train | [link](https://drive.google.com/file/d/1N6vpDBPoHdzkH2SFWPmg4bzVglzmhCMY/view?usp=drive_link)  |
|  Enron Spam | test  | [link](https://drive.google.com/file/d/1LrTFnTKkNDs6FHvQLfmZOTZRUb2Yq0oW/view?usp=drive_link)  |
|  Ag News | train | [link](https://drive.google.com/file/d/1r921scZt8Zd8Lj-i_i65aNiHka98nk34/view?usp=drive_link) |
|  Ag News | test  | [link](https://drive.google.com/file/d/1adpi7n-_gagQ1BULLNsHoUbb0zbb-kX6/view?usp=drive_link) |
|  MIND    | all | [link](https://drive.google.com/file/d/1pq_1kIe2zqwZAhHuROtO-DX_c36__e7J/view?usp=drive_link) |


### Counting word frequency
The pre-computed word count file is [here](https://drive.google.com/file/d/1YrSkDoQL7ComIBr7wYkl1muqZsWSYC2t/view?usp=drive_link).
You can also preprocess wikitext dataset to get the same file.
```bash
cd preparation
python word_count.py
```

### 

Our code is based on the work of [EmbMarker](https://github.com/yjw1029/EmbMarker)

## Citing

```
@article{shetty2024warden,
  title={WARDEN: Multi-Directional Backdoor Watermarks for Embedding-as-a-Service Copyright Protection},
  author={Shetty, Anudeex and Teng, Yue and He, Ke and Xu, Qiongkai},
  journal={arXiv preprint arXiv:2403.01472},
  year={2024}
}
```

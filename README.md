# KGDMC

## Dataset
* Prepare the FB15K-237 and WN18RR datasets following <https://github.com/nju-websoft/CCA>
* Prepare the images of datasets from <https://pan.baidu.com/s/1cbexBtCwxiXM8MeDUvNpRg>, the extraction code is 4xmv

## Noise generation
* Generate modality noise: `python process_image.py`
* Generate triplet noise following <https://github.com/nju-websoft/CCA>

## Run
1. Initial embedding: `python initial_embedding.py`
2. Obtain relation embedding: `python train_transe.py`
3. Obtain textual entity embedding: `python pretain_bert.py`
4. Obtain visual entity embedding: `python pretain_vit.py`
5. Modality error detection: `python eed.py`
6. Obtain initial score: `python initial_score.py`
7. Triplet error detection: `python kged.py`

# TGN
Tensorflow Implementation of the EMNLP-2018 paper [Temporally Grounding Natural Sentence in Video](https://pdfs.semanticscholar.org/452a/ca244ef62a533d8b46a54c6212fe9fa3ce9a.pdf) by Jingyuan Chen *et al.*

![alt text](method.png)

### Requirements
python 2.7
``` bash
pip install -r requirements.txt
```

### Data Preparation
1. Download Glove word embedding data.
``` shell
cd download/
sh download_glove.sh
```

2. Download dataset features.

TACoS: [BaiduDrive](https://pan.baidu.com/s/1OfaQW9622iGor-qXDWbwLA), [GoogleDrive](https://drive.google.com/file/d/13JLnFhSzi8MPRzOG2Ao_q-J5-T5tewcg/view?usp=sharing)

Charades-STA: [BaiduDrive](https://pan.baidu.com/s/1ODW4JIXfCCIbozPcaD_-UA), [GoogleDrive](https://drive.google.com/file/d/1Df2MUL0o0kpxD1IvCCtpazukt4fdMX0a/view?usp=sharing)

ActivityNet-Captions: [BaiduDrive](https://pan.baidu.com/s/1W9S7_nHf3nzDm1TDjm0YBA), [GoogleDrive](https://drive.google.com/file/d/1pkIDiKzzLJwgzkDFumIeOrssLxJhSmox/view?usp=sharing)

Put the feature hdf5 file in the corresponding directory `./datasets/{DATASET}/features/`

We decode TACoS/Charades videos using `fps=16` and extract C3D (fc6) features for each non-overlap 16-frame snippet. Therefore, each feature corresponds to 1-second snippet. For ActivityNet, each feature corresponds to 2-second snippet.

3. Download trained models.

Download and put the checkpoints in corresponding `./checkpoints/{DATASET}/` .

[BaiduDrive](https://pan.baidu.com/s/1A6BczpLZE5I8Jz_6zwnezQ), [GoogleDrive](https://drive.google.com/drive/folders/1iGNNJciF1MzHTtIXNk5MtryLGu5OWNtJ?usp=sharing)


4. Data Preprocessing (Optional)
``` shell
cd datasets/tacos/
sh prepare_data.sh
```
Then copy the generated data in `./data/save/` .

Use correspondig scripts for preparing data for other datasets.

You may skip this procedure as the prepared data is already saved in `./datasets/{DATASET}/data/save/` .

### Testing and Evaluation

``` shell
sh scripts/test_tacos.sh
sh scripts/eval_tacos.sh
```
Use corresponding scripts for testing or evaluating for other datasets.

The predicted results are also provided in `./results/{DATASET}/` .

### Training

``` shell
sh scripts/train_tacos.sh
```
Use corresponding scripts for training for other datasets.
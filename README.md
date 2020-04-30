## iclassifier

reference pytorch code for intent(sentence) classification.
- embedding
  - Glove, BERT, SpanBERT, ALBERT, ROBERTa, BART, ELECTRA
- encoding
  - CNN
  - DenseNet
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
    - implementation from [ntagger](https://github.com/dsindex/ntagger)
  - DSA(Dynamic Self Attention)
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
  - CLS
    - classified by '[CLS]' only for BERT-like architectures. 
- decoding
  - Softmax
- related: [reference pytorch code for entity tagging](https://github.com/dsindex/ntagger)

## requirements

- python >= 3.6

- pip install -r requirements.txt

- pretrained embedding
  - glove
    - [download Glove6B](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embeddings' dir
  ```
  $ mkdir embeddings
  $ ls embeddings
  glove.6B.zip
  $ unzip glove.6B.zip 
  ```
  - BERT, ALBERT, RoBERTa, BART, ELECTRA(huggingface's [transformers](https://github.com/huggingface/transformers.git))
  - [SpanBERT](https://github.com/facebookresearch/SpanBERT/blob/master/README.md)
    - pretrained SpanBERT models are compatible with huggingface's BERT modele except `'bert.pooler.dense.weight', 'bert.pooler.dense.bias'`.

- data
  - Snips
    - `data/snips`
    - from [joint-intent-classification-and-slot-filling-based-on-BERT](https://github.com/lytum/joint-intent-classification-and-slot-filling-based-on-BERT)
      - paper : [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf)
        - intent classification accuracy : **98.6%** (test set)
    - [previous SOTA on SNIPS data](https://paperswithcode.com/sota/intent-detection-on-snips)
  - SST-2
    - `data/sst2`
    - from [GLUE benchmark data](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py)
      - `test.txt` from [pytorch-sentiment-classification](https://github.com/clairett/pytorch-sentiment-classification)
  - TCCC
    - [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)

## Snips data

### experiments summary

|                     | Accuracy (%) | GPU/CPU                     | ONNX     | CONDA             | CONDA+je          | INTEL   | INTEL+je  | Dynamic           | Dynamic+je        | Inference | Inference+Dynamic | Etc            |
| ------------------- | ------------ | --------------------------- | -------- | ----------------- | ----------------- | ------- | --------- | ----------------- | ----------------- | --------- | ----------------- | -------------- |    
| Glove, CNN          | 97.86        | 1.7939  / 4.1414            | 7.5656   | 4.6868  / 3.5353  |                   |         |           |                   |                   | 2.6565    |                   | threads=14     |
| Glove, Densenet-CNN | 97.57        | 3.6094  / 8.3535            | 19.1212  | 7.6969  / 6.9595  |                   |         |           |                   |                   | 6.1414    |                   | threads=14     |
| Glove, Densenet-DSA | 97.43        | 7.5007  / -                 |          |                   |                   |         |           |                   |                   |           |                   |                |
| BERT-base, CNN      | 97.57        | 12.1273 / -                 |          |         / 81.8787 |                   |         |           |         / 52.4949 |                   | 34.7878   | 30.5454           |                |
| BERT-base, CLS      | 97.43        | 12.7714 / 100.929 / 63.7373 | 174.2222 | 69.4343 / 62.5959 | 66.1212 / 63.0707 | 68.9191 | 66        | 66.9494 / 49.4747 | 60.7777 / 50.4040 | 30.7979   | 24.5353           | threads=14     |
| BERT-base, CLS      | 97.00        | 9.2660  / 73.1010 / 43.0707 | 113.2424 | 47.2323 / 43.7070 | 45      / 43.2020 | 48.5050 | 45.2727   | 44.8080 / 34.6565 | 40.8888 / 34.0606 | 19.0707   | 16.1414           | del 8,9,19,11, threads=14 |
| BERT-large, CNN     | **98.00**    | 24.277  / -                 |          |                   |                   |         |           |                   |                   |           |                   |                |
| BERT-large, CLS     | 97.86        | 23.542  / -                 |          |                   |                   |         |           |                   |                   |           |                   |                |

```
* GPU/CPU : Elapsed time/example(ms), GPU / CPU(pip 1.2.0) / CPU(pip 1.5.0)
* ONNX : onnxruntime
* CONDA : conda pytorch=1.2.0/pytorch=1.5.0
* CONDA+je : pytorch=1.2.0/pytorch=1.5.0, etc/jemalloc_omp_kmp.sh
* INTEL : conda pytorch=1.2.0, [intel optimzaed transformers](https://github.com/mingfeima/transformers/tree/kakao/gpt2)
* INTEL+je : conda pytorch=1.2.0, [intel optimzaed transformers](https://github.com/mingfeima/transformers/tree/kakao/gpt2), etc/jemalloc_omp_kmp.sh
* Dynamic : conda pytorch=1.4.0/pytorch=1.5.0, dynamic quantization
* Dynamic+je : conda pytorch=1.4.0/pytorch=1.5.0, dynamic quantization, etc/jemalloc_omp_kmp.sh
* Inference : conda pytorch=1.5.0, --enable_inference
* Inference+Dynamic : conda pytorch=1.5.0, dynamic quantization, --enable_inference
* default batch size, learning rate : 128, 2e-4
```

<details><summary><b>emb_class=glove, enc_class=cnn</b></summary>
<p>
  
- train
```
* token_emb_dim in configs/config-glove-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py --lr_decay_rate=0.9 --embedding_trainable

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py
INFO:__main__:[Accuracy] : 0.9786,   685/  700
INFO:__main__:[Elapsed Time] : 1351ms, 1.793991416309013ms on average
```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet-cnn</b></summary>
<p>
  
- train
```
* token_emb_dim in configs/config-densenet-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-cnn.json
$ python train.py --config=configs/config-densenet-cnn.json --lr_decay_rate=0.9 --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json

INFO:__main__:[Accuracy] : 0.9757,   683/  700
INFO:__main__:[Elapsed Time] : 2633ms, 3.609442060085837ms on average
```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet-dsa</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-densenet-dsa.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-dsa.json
$ python train.py --config=configs/config-densenet-dsa.json --lr_decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 5367ms, 7.500715307582261ms on average
```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cnn
$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

* enc_class=cls
$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

* --bert_use_feature_based for feature-based
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-bert-cnn.json --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9757,   683/  700
INFO:__main__:[Elapsed Time] : 10624ms, 12.127324749642346ms on average
  
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5 , without --bert_do_lower_case
  INFO:__main__:[Accuracy] : 0.9800,   686/  700
  INFO:__main__:[Elapsed Time] : 16994ms, 24.277142857142856ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 8940ms, 12.771428571428572ms on average
  
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5 , without --bert_do_lower_case
  INFO:__main__:[Accuracy] : 0.9786,   685/  700
  INFO:__main__:[Elapsed Time] : 16480ms, 23.542857142857144ms on average

  ** --bert_remove_layers=8,9,10,11 
  INFO:__main__:[Accuracy] : 0.9700,   679/  700
  INFO:__main__:[Elapsed Time] : 6911ms, 9.266094420600858ms on average

```

</p>
</details>


## SST-2 data

### experiments summary

- iclassifier

|                      | Accuracy (%) | GPU/CPU                     | CONDA             | CONDA+je          | Dynamic                  | Dynamic+je        | Etc           |
| -------------------- | ------------ | --------------------------- | ----------------- | ----------------- | ------------------------ | ----------------- | ------------- |
| Glove, CNN           | 82.81        | 1.7670  / 3.9191  / 4.5757  |       - / 4.3131  |       - / 4.4040  |              - / 4.8686  |       - / 4.4848  | threads=14    |
| Glove, DenseNet-CNN  | 86.38        | 3.6203  / 7.1414            |                   |                   |                          |                   | threads=14    |
| Glove, DenseNet-DSA  | 85.34        | 6.2450  / -                 |                   |                   |                          |                   |               |
| BERT-tiny, CNN       | 79.08        | 4.8604  / -                 |                   |                   |                          |                   |               |
| BERT-tiny, CLS       | 80.83        | 3.8461  / -                 |                   |                   |                          |                   |               |
| BERT-mini, CNN       | 83.36        | 7.0983  / -                 |                   |                   |                          |                   |               |
| BERT-mini, CLS       | 83.69        | 5.5521  / -                 |                   |                   |                          |                   |               |
| BERT-small, CNN      | 87.53        | 7.2010  / -                 |                   |                   |                          |                   |               |
| BERT-small, CLS      | 87.86        | 6.0450  / -                 |                   |                   |                          |                   |               |
| BERT-medium, CNN     | 88.58        | 11.9082 / -                 |                   |                   |                          |                   |               |
| BERT-medium, CLS     | 89.24        | 9.5857  / -                 |                   |                   |                          |                   |               |
| BERT-base, CNN       | 92.04        | 14.1576 / -                 |                   |                   |                          |                   |               |
| BERT-base, CLS       | 92.42        | 12.7549 / 100.555 / 62.5050 | 68.5757 / 66.1818 | 65.1616 / 63.1616 | 66.4545(92.42) / 50.8080 | 60.5656 / 50.4343 | threads=14    |
| BERT-base, CNN       | 90.55        | 10.6824 / -                 |                   |                   |                          |                   | del 8,9,10,11 |
| **BERT-base, CLS**   | 91.49        | 8.7747  / 66.6363 / 42.8989 | 46.6262 / 45.6060 | 45.1313 / 45.5050 | 44.7676(90.61) / 34.3131 | 41.3535 / 34.8686 | del 8,9,10,11, threads=14         |
| BERT-base, CLS       | 90.23        | 7.0241  / 51.7676           | 43.5959           |                   |                          |                   | del 6,7,8,9,10,11, threads=14     |
| BERT-base, CLS       | 86.66        | 5.8868  / 36.2121           | 26.5555           |                   |                          |                   | del 4,5,6,7,8,9,10,11, threads=14 |
| BERT-large, CNN      | 93.08        | 28.6490 / -       |            |          |                |            |               |
| BERT-large, CLS      | 93.85        | 27.9967 / -       |            |          |                |            |               |
| BERT-large, CNN      | 88.47        | 14.7813 / -       |            |          |                |            | del 12~23     |
| BERT-large, CLS      | 86.71        | 12.1560 / -       |            |          |                |            | del 12~23     |
| SpanBERT-base, CNN   | 91.82        | 15.2098 / -       |            |          |                |            |               |
| SpanBERT-base, CLS   | 91.49        | 13.1516 / -       |            |          |                |            |               |
| SpanBERT-large, CNN  | 93.90        | 26.8609 / -       |            |          |                |            |               |
| SpanBERT-large, CLS  | 93.96        | 26.0445 / -       |            |          |                |            |               |
| ALBERT-base, CNN     | 92.04        | 16.0554 / -       |            |          |                |            |               |
| ALBERT-base, CLS     | 90.01        | 14.6725 / -       |            |          |                |            |               |
| ALBERT-xxlarge, CNN  | 95.77        | 57.4631 / -       |            |          |                |            |               |
| ALBERT-xxlarge, CLS  | 94.45        | 51.8027 / -       |            |          |                |            |               |
| ROBERTa-base, CNN    | 92.92        | 15.1016 / -       |            |          |                |            |               |
| ROBERTa-base, CLS    | 93.03        | 14.6736 / -       |            |          |                |            |               |
| ROBERTa-base, CNN    | 92.26        | 11.5241 / -       |            |          |                |            | del 8,9,10,11 |
| ROBERTa-base, CLS    | 91.76        | 10.0296 / -       |            |          |                |            | del 8,9,10,11 |
| ROBERTa-large, CNN   | 95.55        | 26.9807 / -       |            |          |                |            |               |
| ROBERTa-large, CLS   | 95.66        | 23.7395 / -       |            |          |                |            |               |
| BART-large, CNN      | 94.45        | 35.1708 / -       |            |          |                |            |               |
| BART-large, CLS      | 94.89        | 33.3862 / -       |            |          |                |            |               |
| ELECTRA-base, CNN    | 95.39        | 14.9802 / -       |            |          |                |            |               |
| ELECTRA-base, CLS    | 95.22        | 14.0087 / -       |            |          |                |            |               |
| ELECTRA-large, CNN   | 96.05        | 27.2868 / -       |            |          |                |            |               |
| ELECTRA-large, CLS   | **96.43**    | 25.6857 / -       |            |          |                |            |               |

- [sst2 leaderboard](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary)

|                   | Accuracy (%)|
| ----------------- | ----------- |
| T5-3B             | 97.4        |
| ALBERT            | 97.1        |
| RoBERTa           | 96.7        |
| MT-DNN            | 95.6        |
| DistilBERT        | 92.7        |

<details><summary><b>emb_class=glove, enc_class=cnn</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/sst2
$ python train.py --data_dir=data/sst2 --lr=1e-3 --lr_decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8281,  1508/ 1821
INFO:__main__:[Elapsed Time] : 3300ms, 1.767032967032967ms on average

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet-cnn</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-densenet-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --lr_decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8638,  1573/ 1821
INFO:__main__:[Elapsed Time] : 6678ms, 3.6203296703296703ms on average
```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet-dsa</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-densenet-dsa.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --lr_decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8534,  1554/ 1821
INFO:__main__:[Elapsed Time] : 11459ms, 6.245054945054945ms on average

* try again
INFO:__main__:[Accuracy] : 0.8506,  1549/ 1821
INFO:__main__:[Elapsed Time] : 21745ms, 11.885714285714286ms on average

* softmax masking
INFO:__main__:[Accuracy] : 0.8473,  1543/ 1821
INFO:__main__:[Elapsed Time] : 19214ms, 10.477472527472527ms on average
```

</p>
</details>


<details><summary><b>emb_class=bert, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cnn
$ python preprocess.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64

* enc_class=cls
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9204,  1676/ 1821
INFO:__main__:[Elapsed Time] : 25878ms, 14.157692307692308ms on average

  ** --bert_model_name_or_path=bert-large-uncased
  INFO:__main__:[Accuracy] : 0.9308,  1695/ 1821
  INFO:__main__:[Elapsed Time] : 52170ms, 28.649093904448105ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-8_H-512_A-8
  INFO:__main__:[Accuracy] : 0.8858,  1613/ 1821
  INFO:__main__:[Elapsed Time] : 21791ms, 11.908241758241758ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-4_H-512_A-8
  INFO:__main__:[Accuracy] : 0.8753,  1594/ 1821
  INFO:__main__:[Elapsed Time] : 13206ms, 7.201098901098901ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-4_H-256_A-4
  INFO:__main__:[Accuracy] : 0.8336,  1518/ 1821
  INFO:__main__:[Elapsed Time] : 13021ms, 7.098351648351648ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-2_H-128_A-2
  INFO:__main__:[Accuracy] : 0.7908,  1440/ 1821
  INFO:__main__:[Elapsed Time] : 8951ms, 4.86043956043956ms on average

  ** for using SpanBERT embedding, just replace pretrained BERT model to SpanBERT.
  ** --bert_model_name_or_path=embeddings/spanbert_hf_large , without --bert_do_lower_case
  INFO:__main__:[Accuracy] : 0.9390,  1710/ 1821
  INFO:__main__:[Elapsed Time] : 49042ms, 26.860989010989012ms on average

  ** --bert_model_name_or_path=embeddings/spanbert_hf_base , without --bert_do_lower_case
  INFO:__main__:[Accuracy] : 0.9182,  1672/ 1821
  INFO:__main__:[Elapsed Time] : 27796ms, 15.20989010989011ms on average

  ** --bert_remove_layers=8,9,10,11
  INFO:__main__:[Accuracy] : 0.9055,  1649/ 1821
  INFO:__main__:[Elapsed Time] : 19541ms, 10.682417582417582ms on average

  ** --bert_model_name_or_path=bert-large-uncased --bert_remove_layers=12,13,14,15,16,17,18,19,20,21,22,23 
  INFO:__main__:[Accuracy] : 0.8847,  1611/ 1821
  INFO:__main__:[Elapsed Time] : 27017ms, 14.781318681318682ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9242,  1683/ 1821
INFO:__main__:[Elapsed Time] : 23314ms, 12.754945054945056ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9385,  1709/ 1821
  INFO:__main__:[Elapsed Time] : 50982ms, 27.99670510708402ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-8_H-512_A-8
  INFO:__main__:[Accuracy] : 0.8924,  1625/ 1821
  INFO:__main__:[Elapsed Time] : 17558ms, 9.585714285714285ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-4_H-512_A-8
  INFO:__main__:[Accuracy] : 0.8786,  1600/ 1821
  INFO:__main__:[Elapsed Time] : 11104ms, 6.045054945054945ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-4_H-256_A-4
  INFO:__main__:[Accuracy] : 0.8369,  1524/ 1821
  INFO:__main__:[Elapsed Time] : 10196ms, 5.552197802197802ms on average

  ** --bert_model_name_or_path=embeddings/pytorch.uncased_L-2_H-128_A-2
  INFO:__main__:[Accuracy] : 0.8083,  1472/ 1821
  INFO:__main__:[Elapsed Time] : 7124ms, 3.8461538461538463ms on average

  ** for using SpanBERT embedding, just replace pretrained BERT model to SpanBERT.
  ** --bert_model_name_or_path=embeddings/spanbert_hf_large , without --bert_do_lower_case
  INFO:__main__:[Accuracy] : 0.9396,  1711/ 1821
  INFO:__main__:[Elapsed Time] : 47570ms, 26.044505494505493ms on average

  ** --bert_model_name_or_path=embeddings/spanbert_hf_base , without --bert_do_lower_case
  INFO:__main__:[Accuracy] : 0.9149,  1666/ 1821
  INFO:__main__:[Elapsed Time] : 24049ms, 13.151648351648351ms on average

  ** --bert_remove_layers=8,9,10,11
  INFO:__main__:[Accuracy] : 0.9149,  1666/ 1821
  INFO:__main__:[Elapsed Time] : 16082ms, 8.774725274725276ms on average

  ** --bert_remove_layers=6,7,8,9,10,11
  INFO:__main__:[Accuracy] : 0.9023,  1643/ 1821
  INFO:__main__:[Elapsed Time] : 12865ms, 7.024175824175824ms on average

  ** --bert_remove_layers=4,5,6,7,8,9,10,11
  INFO:__main__:[Accuracy] : 0.8666,  1578/ 1821
  INFO:__main__:[Elapsed Time] : 10800ms, 5.886813186813187ms on average

  ** --bert_model_name_or_path=bert-large-uncased --bert_remove_layers=12,13,14,15,16,17,18,19,20,21,22,23 
  INFO:__main__:[Accuracy] : 0.8671,  1579/ 1821
  INFO:__main__:[Elapsed Time] : 22229ms, 12.156043956043955ms on average

```

</p>
</details>


<details><summary><b>emb_class=albert, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cnn
$ python preprocess.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2 --bert_do_lower_case
$ python train.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2 --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64 --bert_do_lower_case
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint 

INFO:__main__:[Accuracy] : 0.9204,  1676/ 1821
INFO:__main__:[Elapsed Time] : 29321ms, 16.055494505494504ms on average

  ** --bert_model_name_or_path=./embeddings/albert-xxlarge-v2 --batch_size=32
  INFO:__main__:[Accuracy] : 0.9577,  1744/ 1821
  INFO:__main__:[Elapsed Time] : 104769ms, 57.463186813186816ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-albert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9001,  1639/ 1821
INFO:__main__:[Elapsed Time] : 26819ms, 14.672527472527472ms on average

  ** --bert_model_name_or_path=./embeddings/albert-xxlarge-v2 --batch_size=32
  INFO:__main__:[Accuracy] : 0.9445,  1720/ 1821
  INFO:__main__:[Elapsed Time] : 94456ms, 51.80274725274725ms on average

```

</p>
</details>


<details><summary><b>emb_class=roberta, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cnn
$ python preprocess.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large
$ python train.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64

* enc_class=cls
$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large 
$ python train.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9555,  1740/ 1821
INFO:__main__:[Elapsed Time] : 49297ms, 26.98076923076923ms on average

  ** --bert_model_name_or_path=./embeddings/roberta-base
  INFO:__main__:[Accuracy] : 0.9292,  1692/ 1821
  INFO:__main__:[Elapsed Time] : 27615ms, 15.101648351648352ms on average

  ** --bert_model_name_or_path=./embeddings/roberta-base --bert_remove_layers=8,9,10,11
  INFO:__main__:[Accuracy] : 0.9226,  1680/ 1821
  INFO:__main__:[Elapsed Time] : 21127ms, 11.524175824175824ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9566,  1742/ 1821
INFO:__main__:[Elapsed Time] : 43363ms, 23.73956043956044ms on average

  ** --bert_model_name_or_path=./embeddings/roberta-base
  INFO:__main__:[Accuracy] : 0.9303,  1694/ 1821
  INFO:__main__:[Elapsed Time] : 26822ms, 14.673626373626373ms on average

  ** --bert_model_name_or_path=./embeddings/roberta-base --bert_remove_layers=8,9,10,11
  INFO:__main__:[Accuracy] : 0.9176,  1671/ 1821
  INFO:__main__:[Elapsed Time] : 18344ms, 10.02967032967033ms on average

```

</p>
</details>


<details><summary><b>emb_class=bart, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cnn
$ python preprocess.py --config=configs/config-bart-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large
$ python train.py --config=configs/config-bart-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64

* enc_class=cls
$ python preprocess.py --config=configs/config-bart-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large 
$ python train.py --config=configs/config-bart-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-bart-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9445,  1720/ 1821
INFO:__main__:[Elapsed Time] : 64224ms, 35.17087912087912ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-bart-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9489,  1728/ 1821
INFO:__main__:[Elapsed Time] : 61015ms, 33.386263736263736ms on average

```

</p>
</details>


<details><summary><b>emb_class=electra, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cnn
$ python preprocess.py --config=configs/config-electra-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_do_lower_case
$ python train.py --config=configs/config-electra-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64 --bert_do_lower_case

* enc_class=cls
$ python preprocess.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_do_lower_case
$ python train.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64 --bert_do_lower_case
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-electra-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint 

INFO:__main__:[Accuracy] : 0.9539,  1737/ 1821
INFO:__main__:[Elapsed Time] : 29602ms, 14.98021978021978ms on average

  ** --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6
  INFO:__main__:[Accuracy] : 0.9566,  1742/ 1821
  INFO:__main__:[Elapsed Time] : 54157ms, 28.356593406593408ms on average

  ** --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6 --epoch=15
  INFO:__main__:[Accuracy] : 0.9605,  1749/ 1821
  INFO:__main__:[Elapsed Time] : 52163ms, 27.286813186813188ms on average

* enc_lass=cls
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9522,  1734/ 1821
INFO:__main__:[Elapsed Time] : 25956ms, 14.008791208791209ms on average

  ** --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6 --epoch=15
  INFO:__main__:[Accuracy] : 0.9643,  1756/ 1821
  INFO:__main__:[Elapsed Time] : 47163ms, 25.685714285714287ms on average
 
```

</p>
</details>

## experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

## optimization

- [OPTIMIZATION.md](/OPTIMIZATION.md)

## serving

- [Deploying huggingface‘s BERT to production with pytorch/serve](https://medium.com/@freidankm_39840/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)

## references

- [Intent Detection](https://paperswithcode.com/task/intent-detection)
- [Intent Classification](https://paperswithcode.com/task/intent-classification)
- [Identifying Hate Speech with BERT and CNN](https://towardsdatascience.com/identifying-hate-speech-with-bert-and-cnn-b7aa2cddd60d)
- [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta)
  - [RoBERTa GLUE task setting](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.glue.md)
- [BERT Miniatures](https://huggingface.co/google/bert_uncased_L-12_H-128_A-2)
  - search range of hyperparameters
    - batch sizes: 8, 16, 32, 64, 128
    - learning rates: 3e-4, 1e-4, 5e-5, 3e-5
- scalar mixtures of BERT all layers
  - [ScalarMixWithDropout](https://github.com/Hyperparticle/udify/blob/master/udify/modules/scalar_mix.py)
  - [ScalarMix](https://github.com/allenai/allennlp/blob/master/allennlp/modules/scalar_mix.py)
- [Poor Man’s BERT: Smaller and Faster Transformer Models](https://arxiv.org/pdf/2004.03844v1.pdf)
  - https://github.com/hsajjad/transformers/blob/master/examples/run_glue.py

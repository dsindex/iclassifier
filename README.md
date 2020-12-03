# Description

**reference pytorch code for intent(sentence) classification.**

- embedding
  - GloVe, BERT, DistilBERT, SpanBERT, ALBERT, RoBERTa, BART, ELECTRA
- encoding
  - GNB
    - Gaussian Naive Bayes(simple biased model)
  - CNN
    - Convolutional Neural Net)
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

<br>



# Requirements

- python >= 3.6

- pip install -r requirements.txt

- pretrained embedding
  - glove
    - [download GloVe6B](http://nlp.stanford.edu/data/glove.6B.zip)
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

<br>



# Data

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
    - pending

<br>


# Snips data

### experiments summary

|                      | Accuracy (%) | GPU / CPU           | ONNX                         | CONDA                       | Dynamic                     | Inference         | Inference+Dynamic | Inference+ONNX                 | Etc            |
| -------------------- | ------------ | ------------------- | ---------------------------- | --------------------------- | --------------------------- | ----------------- | ----------------- | ------------------------------ | -------------- |    
| GloVe, GNB           | 80.43        | 1.2929  / -         | -        / -       / -       | -       / -                 |                             | -                 |                   | -                              |                |
| GloVe, CNN           | 97.86        | 1.7939  / -         | 7.5656   / 1.8689  / 1.7735  | 4.6868  / 2.7592            |                             | 1.9398            |                   | 0.3848  / -       / FAILED     | threads=14     |
| GloVe, Densenet-CNN  | 97.57        | 3.6094  / -         | 19.1212  / 3.0717  / 3.0917  | 7.6969  / 6.5887            |                             | 4.9481            |                   | 0.8658  / -       / FAILED     | threads=14     |
| GloVe, Densenet-DSA  | 97.43        | 7.5007  / -         | -        / 4.4936  / 4.9337  |         / 9.7873            |                             | 7.2086            |                   | 1.5420  / -       / FAILED     | threads=14     |
| BERT-small, CLS      | 98.00        | 5.9837  / -         | -        / -       / 12.0953 | -       / -       / 17.4995 | -       / -       / 15.2820 | -       / 7.4538  | -       / 7.2436  | -       / 3.5445  / **2.4141** | threads=14     |
| DistilBERT, CLS      | 97.71        | 9.3075  / -         | -        / 32.4263 / 31.1101 | -       / 37.7777           | -       / 29.3939           | 14.9494           | 10.4040           | 8.9942  / 10.1848 / 4.8818     | threads=14     |
| SqueezeBERT, CLS     | -            | 18.0796 / -         | -        / -       / -       | -       / -       / 24.0667 | -       / -       / 23.8565 | -       / 20.3999 | -       / 20.0118 | -       / 11.9890 / FAILED     | threads=14     |
| BERT-base, CNN       | 97.57        | 12.1273 / -         |                              | -       / 81.8787           | -       / -                 | 34.7878           | 30.5454           |                                | threads=14     |
| BERT-base, CLS       | 97.43        | 12.7714 / 100.929   | 174.2222 / 46.4263 / 43.5078 | 69.4343 / 62.5959           | 66.9494 / 49.4747           | 30.7979           | 24.5353           | 16.9756                        | threads=14     |
| BERT-base, CLS       | 97.00        | 9.2660  / 73.1010   | 113.2424 / 31.5400 / 26.9472 | 47.2323 / 42.8950           | 44.8080 / 33.4623           | 16.7419           | 13.5703           | 11.7487                        | del 8,9,19,11, threads=14 |
| BERT-large, CNN      | **98.00**    | 24.277  / -         |                              |                             |                             |                   |                   |                                |                |
| BERT-large, CLS      | 97.86        | 23.542  / -         |                              |                             |                             |                   |                   |                                |                |

```
* GPU / CPU : Elapsed time/example(ms), GPU / CPU(pip 1.2.0)  [Tesla V100 1 GPU, Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 2 CPU, 14CORES/1CPU, HyperThreading]
* ONNX : onnxruntime 1.2.0, pip pytorch==1.2.0 
         / onnxruntime 1.3.0, conda pytorch=1.5.0 
         / onnxruntime 1.3.0, conda pytorch=1.5.0, onnxruntime_tools.optimizer_cli
* CONDA : conda pytorch=1.2.0
          / conda pytorch=1.5.0
          / conda pytorch=1.6.0
* Dynamic : conda pytorch=1.4.0, --enable_dqm
            / conda pytorch=1.5.0, --enable_dqm
            / conda pytorch=1.6.0, --enable_dqm
* Inference : conda pytorch=1.5.0, --enable_inference
              / conda pytorch=1.6.0, --enable_inference
* Inference+Dynamic : conda pytorch=1.5.0, --enable_dqm, --enable_inference
                      / conda pytorch=1.6.0, --enable_dqm, --enable_inference 
* Inference+ONNX : conda pytorch=1.5.0, --enable_ort(onnxruntime 1.3.0), --enable_inference
                   / conda pytorch=1.6.0, --enable_ort(onnxruntime 1.4.0), --enable_inference 
                   / conda pytorch=1.6.0, --enable_ort(onnxruntime 1.4.0), --enable_inference, --quantize_onnx
* default batch size, learning rate, n_ctx(max_seq_length) : 128, 2e-4, 100
* number of tokens / sentence : MEAN : 9.08, MAX:24, MIN:3, MEDIAN:9
```

| # threads | Model               | Inference+ONNX / Inference+QuantizedONNX  | Etc              |
| --------- | ------------------- | ----------------------------------------- | ---------------- |
| 1         | GloVe, Densenet-DSA | 3.33    /  -                              |                  |
| 1         | DistilBERT, CLS     | 51.77   /  15.66                          |                  |
| 2         | DistilBERT, CLS     | 28.38   /  9.71                           |                  |
| 3         | DistilBERT, CLS     | 21.47   /  7.79                           |                  |
| 4         | DistilBERT, CLS     | 18.75   /  6.69                           |                  |
| 5         | DistilBERT, CLS     | 15.23   /  6.09                           |                  |
| 6         | DistilBERT, CLS     | 14.22   /  5.69                           |                  |
| 7         | DistilBERT, CLS     | 12.52   /  5.44                           |                  |
| 8         | DistilBERT, CLS     | 10.46   /  5.21                           | **good enough**  |
| 9         | DistilBERT, CLS     | 10.93   /  5.17                           |                  |
| 10        | DistilBERT, CLS     | 9.75    /  4.99                           |                  |
| 11        | DistilBERT, CLS     | 9.22    /  4.98                           |                  |
| 12        | DistilBERT, CLS     | 10.11   /  4.91                           |                  |
| 13        | DistilBERT, CLS     | 9.45    /  4.81                           |                  |
| 14        | DistilBERT, CLS     | 9.31    /  4.74                           |                  |

<details><summary><b>emb_class=glove, enc_class=gnb</b></summary>
<p>
  
- train
```
* token_emb_dim in configs/config-glove-gnb.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-glove-gnb.json
$ python train.py --config=configs/config-glove-gnb.json
```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-gnb.json
INFO:__main__:[Accuracy] : 0.8043,   563/  700
INFO:__main__:[Elapsed Time] : 980.9308052062988ms, 1.292972264542259ms on average

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=cnn</b></summary>
<p>
  
- train
```
* token_emb_dim in configs/config-glove-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-glove-cnn.json
$ python train.py --config=configs/config-glove-cnn.json --embedding_trainable

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-cnn.json
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
$ python train.py --config=configs/config-densenet-cnn.json --embedding_trainable
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
$ python train.py --config=configs/config-densenet-dsa.json
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

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/bert-base-uncased
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9757,   683/  700
INFO:__main__:[Elapsed Time] : 10624ms, 12.127324749642346ms on average
  
** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
INFO:__main__:[Accuracy] : 0.9800,   686/  700
INFO:__main__:[Elapsed Time] : 16994ms, 24.277142857142856ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 8940ms, 12.771428571428572ms on average
  
** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
INFO:__main__:[Accuracy] : 0.9786,   685/  700
INFO:__main__:[Elapsed Time] : 16480ms, 23.542857142857144ms on average

** --bert_remove_layers=8,9,10,11 
INFO:__main__:[Accuracy] : 0.9700,   679/  700
INFO:__main__:[Elapsed Time] : 6911ms, 9.266094420600858ms on average

** --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/distilbert-base-uncased
INFO:__main__:[Accuracy] : 0.9771,   684/  700
INFO:__main__:[Elapsed Time] : 6607ms, 9.30758226037196ms on average

** --config=configs/config-funnel-cls.json --bert_model_name_or_path=./embeddings/funnel-transformer-small-base
INFO:__main__:[Accuracy] : 0.9629,   674/  700
INFO:__main__:[Elapsed Time] : 14707.713603973389ms, 20.878405018425806ms on average

** --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/squeezebert-uncased
INFO:__main__:[Accuracy] : 0.9729,   681/  700
INFO:__main__:[Elapsed Time] : 12742.885112762451ms, 18.07960045013646ms on average

** --config=configs/config-bert-cls.json --bert_model_name_or_paht=./embeddings/pytorch.uncased_L-4_H-512_A-8
INFO:__main__:[Accuracy] : 0.9800,   686/  700
INFO:__main__:[Elapsed Time] : 4279.639005661011ms, 5.983798800619887ms on average

```

</p>
</details>

<br>



# SST-2 data

### experiments summary

- iclassifier

|                                         | Accuracy (%) | GPU/CPU                     | CONDA                          | Dynamic                  | Etc           |
| --------------------------------------- | ------------ | --------------------------- | ------------------------------ | ------------------------ | ------------- |
| GloVe, GNB                              | 72.27        | 1.2253  / -       / -       |       - / -                    |              - / -       | -             |
| GloVe, CNN                              | 82.81        | 1.7670  / 3.9191  / 4.5757  |       - / 4.3131               |              - / 4.8686  | threads=14    |
| ConceptNet, CNN                         | 84.79        | 2.8304  / -       / -       |       - / -                    |              - / -       |               |
| ConceptNet, CNN                         | 84.90        | 2.7672  / -       / -       |       - / -                    |              - / -       | optuna        |
| GloVe, DenseNet-CNN                     | 86.38        | 3.6203  / 7.1414            |                                |                          | threads=14    |
| ConceptNet, DenseNet-CNN                | 87.26        | 3.7052  / -                 |                                |                          |               |
| ConceptNet, DenseNet-CNN                | 87.26        | 3.5377  / -                 |                                |                          | optuna        |
| GloVe, DenseNet-DSA                     | 85.34        | 6.2450  / -                 |                                |                          |               |
| DistilFromBERT, GloVe, CNN              | 86.16        | 1.7900  / -                 |                                |                          | augmented, from large             |
| DistilFromBERT, GloVe, DenseNet-CNN     | 88.52        | 3.6788  / -                 |                                |                          | augmented, from large             |
| DistilFromBERT, GloVe, DenseNet-DSA     | 88.14        | 8.4647  / -                 |                                |                          | augmented, from large             |
| DistilFromBERT, BERT-small, CLS         | 89.29        | 5.9408  / -                 |                                |                          | fastformers, from base            |
| DistilFromBERT, BERT-small, CLS         | 91.49        | 5.9114  / -                 |                                |                          | fastformers, augmented, from base |
| DistilFromBERT, BERT-small, CLS         | 90.33        | 6.0072  / -                 |                                |                          | fastformers, from large           |
| DistilFromRoBERTa, GloVe, CNN           | 86.55        | 1.8483  / -                 |                                |                          | augmented, from large             |
| DistilFromRoBERTa, GloVe, DenseNet-CNN  | 88.80        | 3.9580  / -                 |                                |                          | augmented, from large             |
| DistilFromRoBERTa, GloVe, DenseNet-DSA  | 88.25        | 8.5627  / -                 |                                |                          | augmented, from large             |
| DistilFromELECTRA, GloVe, CNN           | 86.55        | 1.7466  / -                 |                                |                          | augmented, from large             |
| DistilFromELECTRA, GloVe, DenseNet-CNN  | 89.79        | 3.6406  / -                 |                                |                          | augmented, from large             |
| DistilFromELECTRA, GloVe, DenseNet-DSA  | 88.58        | 8.3708  / -                 |                                |                          | augmented, from large             |
| DistilFromELECTRA, DistilBERT, CLS      | 93.52        | 7.4879  / -                 |                                |                          | augmented, from large             |
| BERT-tiny, CNN                          | 79.08        | 4.8604  / -                 |                                |                          |               |
| BERT-tiny, CLS                          | 80.83        | 3.8461  / -                 |                                |                          |               |
| BERT-mini, CNN                          | 83.36        | 7.0983  / -                 |                                |                          |               |
| BERT-mini, CLS                          | 83.69        | 5.5521  / -                 |                                |                          |               |
| BERT-small, CNN                         | 87.53        | 7.2010  / -                 |                                |                          |               |
| BERT-small, CLS                         | 88.25        | 5.9604  / -                 |                                |                          |               |
| BERT-medium, CNN                        | 88.58        | 11.9082 / -                 |                                |                          |               |
| BERT-medium, CLS                        | 89.24        | 9.5857  / -                 |                                |                          |               |
| DistilBERT, CNN                         | 89.90        | 9.9362  / -                 |       - / 44.1111              |              - / 35.7070 | threads=14    |
| **DistilBERT, CLS**                     | 91.10        | 8.9719  / -                 |       - / 37.2626              |              - / 29.4646 | threads=14    |
| BERT-base, CNN                          | 92.04        | 14.1576 / -                 |                                |                          |               |
| BERT-base, CLS                          | 92.42        | 12.7549 / 100.555 / 62.5050 | 68.5757 / 66.1818              | 66.4545(92.42) / 50.8080 | threads=14    |
| BERT-base, CLS                          | 93.36        | 15.6755 / -                 |       - / -                    |              - / -       | fintuned using amazon reviews     |
| BERT-base, CLS                          | 93.25        | 14.2535 / -                 |       - / -                    |              - / -       | augmented                         |
| BERT-base, CNN                          | 90.55        | 10.6824 / -                 |                                |                          | del 8,9,10,11 |
| BERT-base, CLS                          | 91.49        | 8.7747  / 66.6363 / 42.8989 | 46.6262 / 45.6060              | 44.7676(90.61) / 34.3131 | del 8,9,10,11, threads=14         |
| BERT-base, CLS                          | 90.23        | 7.0241  / 51.7676           | 43.5959                        |                          | del 6,7,8,9,10,11, threads=14     |
| BERT-base, CLS                          | 86.66        | 5.8868  / 36.2121           | 26.5555                        |                          | del 4,5,6,7,8,9,10,11, threads=14 |
| BERT-large, CNN                         | 93.08        | 28.6490 / -                 |                                |                          |               |
| BERT-large, CLS                         | 94.12        | 22.3767 / -                 |                                |                          |               |
| BERT-large, CLS                         | 93.57        | 27.3209 / -                 |                                |                          | fintuned using amazon reviews     |
| BERT-large, CNN                         | 88.47        | 14.7813 / -                 |                                |                          | del 12~23     |
| BERT-large, CLS                         | 86.71        | 12.1560 / -                 |                                |                          | del 12~23     |
| SqueezeBERT, CNN                        | 90.61        | 19.2879 / -                 |       - / -       / 69.5269    |                          | threads=14    |
| SqueezeBERT, CLS                        | 90.12        | 17.4998 / -                 |       - / -       / 63.8998    |                          | threads=14    |
| SpanBERT-base, CNN                      | 91.82        | 15.2098 / -                 |                                |                          |               |
| SpanBERT-base, CLS                      | 91.49        | 13.1516 / -                 |                                |                          |               |
| SpanBERT-large, CNN                     | 93.90        | 26.8609 / -                 |                                |                          |               |
| SpanBERT-large, CLS                     | 93.96        | 26.0445 / -                 |                                |                          |               |
| ALBERT-base, CNN                        | 92.04        | 16.0554 / -                 |                                |                          |               |
| ALBERT-base, CLS                        | 90.01        | 14.6725 / -                 |                                |                          |               |
| ALBERT-xxlarge, CNN                     | 95.77        | 57.4631 / -                 |                                |                          |               |
| ALBERT-xxlarge, CLS                     | 94.45        | 51.8027 / -                 |                                |                          |               |
| RoBERTa-base, CNN                       | 92.92        | 15.1016 / -                 |                                |                          |               |
| RoBERTa-base, CLS                       | 93.03        | 14.6736 / -                 |                                |                          |               |
| RoBERTa-base, CNN                       | 92.26        | 11.5241 / -                 |                                |                          | del 8,9,10,11 |
| RoBERTa-base, CLS                       | 91.76        | 10.0296 / -                 |                                |                          | del 8,9,10,11 |
| RoBERTa-large, CNN                      | 95.55        | 26.9807 / -                 |                                |                          |               |
| RoBERTa-large, CLS                      | 95.66        | 23.7395 / -                 |                                |                          |               |
| BART-large, CNN                         | 94.45        | 35.1708 / -                 |                                |                          |               |
| BART-large, CLS                         | 94.89        | 33.3862 / -                 |                                |                          |               |
| ELECTRA-base, CNN                       | 95.39        | 14.9802 / -                 |                                |                          |               |
| ELECTRA-base, CLS                       | 95.22        | 14.0087 / -                 |                                |                          |               |
| ELECTRA-large, CNN                      | 96.05        | 27.2868 / -                 |                                |                          |               |
| ELECTRA-large, CLS                      | **96.43**    | 25.6857 / -                 |                                |                          |               |

- [sst2 leaderboard](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary)

|                   | Accuracy (%)|
| ----------------- | ----------- |
| T5-3B             | 97.4        |
| ALBERT            | 97.1        |
| RoBERTa           | 96.7        |
| MT-DNN            | 95.6        |
| DistilBERT        | 92.7        |


<details><summary><b>emb_class=glove, enc_class=gnb</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove-gnb.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-glove-gnb.json --data_dir=data/sst2
$ python train.py --config=configs/config-glove-gnb.json --data_dir=data/sst2 --lr=1e-3
```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-gnb.json --data_dir=data/sst2
INFO:__main__:[Accuracy] : 0.7227,  1316/ 1821
INFO:__main__:[Elapsed Time] : 2310.748338699341ms, 1.2253080095563615ms on average

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=cnn</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-glove-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/sst2
$ python train.py --data_dir=data/sst2 --lr=1e-3 
```

- evaluation
```
$ python evaluate.py --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8281,  1508/ 1821
INFO:__main__:[Elapsed Time] : 3300ms, 1.767032967032967ms on average

* --embedding_path=./embeddings/numberbatch-en-19.08.txt (from https://github.com/commonsense/conceptnet-numberbatch)
INFO:__main__:[Accuracy] : 0.8479,  1544/ 1821
INFO:__main__:[Elapsed Time] : 5237.588167190552ms, 2.830480088244428ms on average

* --embedding_path=./embeddings/numberbatch-en-19.08.txt --seed 36 --batch_size 32 --lr 0.000685889661509286 (by optuna)
INFO:__main__:[Accuracy] : 0.8490,  1546/ 1821
INFO:__main__:[Elapsed Time] : 5137.487411499023ms, 2.7672590790214118ms on average

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet-cnn</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-densenet-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8638,  1573/ 1821
INFO:__main__:[Elapsed Time] : 6678ms, 3.6203296703296703ms on average

* --embedding_path=./embeddings/numberbatch-en-19.08.txt (from https://github.com/commonsense/conceptnet-numberbatch)
INFO:__main__:[Accuracy] : 0.8726,  1589/ 1821
INFO:__main__:[Elapsed Time] : 6822.043418884277ms, 3.7052182050851674ms on average

* --embedding_path=./embeddings/numberbatch-en-19.08.txt --seed 42 --batch_size 64 --lr 0.0005115118656470668 (by optuna)
INFO:__main__:[Accuracy] : 0.8726,  1589/ 1821
INFO:__main__:[Elapsed Time] : 6442.193614435720869ms, 3.537723017262889ms on average

```

</p>
</details>


<details><summary><b>emb_class=glove, enc_class=densenet-dsa</b></summary>
<p>

- train
```
* token_emb_dim in configs/config-densenet-dsa.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2
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

$ python preprocess.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased
$ python train.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64
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

** --configs/config-distilbert-cnn.json --bert_model_name_or_path=embeddings/distilbert-base-uncased
INFO:__main__:[Accuracy] : 0.8990,  1637/ 1821
INFO:__main__:[Elapsed Time] : 18193ms, 9.936263736263736ms on average

** --bert_model_name_or_path=./embeddings/squeezebert-uncased --epoch=20 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9061,  1650/ 1821
INFO:__main__:[Elapsed Time] : 35230.47494888306ms, 19.287928644117418ms on average

** for using SpanBERT embedding, just replace pretrained BERT model to SpanBERT.
** --bert_model_name_or_path=embeddings/spanbert_hf_large
INFO:__main__:[Accuracy] : 0.9390,  1710/ 1821
INFO:__main__:[Elapsed Time] : 49042ms, 26.860989010989012ms on average

** --bert_model_name_or_path=embeddings/spanbert_hf_base
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

** n_ctx=64
INFO:__main__:[Accuracy] : 0.9259,  1686/ 1821
INFO:__main__:[Elapsed Time] : 23765.23184776306ms, 13.007715246179602ms on average

** n_ctx=64, --lr=2e-5 --epoch=3 --batch_size=64  --warmup_epoch=1 --weight_decay=0.0 --seed=0
INFO:__main__:[Accuracy] : 0.9281,  1690/ 1821
INFO:__main__:[Elapsed Time] : 21707.942724227905ms, 11.878120244204343ms on average

** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
INFO:__main__:[Accuracy] : 0.9412,  1714/ 1821
INFO:__main__:[Elapsed Time] : 40847.62740135193ms, 22.37672412788475ms on average

** --bert_model_name_or_path=embeddings/pytorch.uncased_L-8_H-512_A-8
INFO:__main__:[Accuracy] : 0.8924,  1625/ 1821
INFO:__main__:[Elapsed Time] : 17558ms, 9.585714285714285ms on average

** --bert_model_name_or_path=embeddings/pytorch.uncased_L-4_H-512_A-8
INFO:__main__:[Accuracy] : 0.8825,  1607/ 1821
INFO:__main__:[Elapsed Time] : 10948.646068572998ms, 5.960442338671003ms on average

** --bert_model_name_or_path=embeddings/pytorch.uncased_L-4_H-256_A-4
INFO:__main__:[Accuracy] : 0.8369,  1524/ 1821
INFO:__main__:[Elapsed Time] : 10196ms, 5.552197802197802ms on average

** --bert_model_name_or_path=embeddings/pytorch.uncased_L-2_H-128_A-2
INFO:__main__:[Accuracy] : 0.8083,  1472/ 1821
INFO:__main__:[Elapsed Time] : 7124ms, 3.8461538461538463ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=embeddings/distilbert-base-uncased
INFO:__main__:[Accuracy] : 0.9110,  1659/ 1821
INFO:__main__:[Elapsed Time] : 16431ms, 8.971978021978021ms on average

** --bert_model_name_or_path=./embeddings/squeezebert-uncased --epoch=20  --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9012,  1641/ 1821
INFO:__main__:[Elapsed Time] : 31975.939989089966ms, 17.499822050660523ms on average

** for using SpanBERT embedding, just replace pretrained BERT model to SpanBERT.
** --bert_model_name_or_path=embeddings/spanbert_hf_large
INFO:__main__:[Accuracy] : 0.9396,  1711/ 1821
INFO:__main__:[Elapsed Time] : 47570ms, 26.044505494505493ms on average

** --bert_model_name_or_path=embeddings/spanbert_hf_base
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

** fine-tune bert-base-uncased using amazon_us_reviews data(https://huggingface.co/nlp/viewer/?dataset=amazon_us_reviewss&config=Video_v1_00). and then apply to sst2 data.
$ cd etc
$ python download_datasets.py --dataset_name=amazon_us_reviews --task_name=Video_v1_00 --split=train > ../data/amazon_us_reviews_Video_v1_00/total.txt
$ cd ../data/amazon_us_reviews_Video_v1_00
$ python ../etc/split.py --data_path=total.txt --base_path=data

# we have 'data.train', 'data.valid', 'data.test'.
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/amazon_us_reviews_Video_v1_00 --bert_model_name_or_path=./embeddings/bert-base-uncased
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/amazon_us_reviews_Video_v1_00 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint-amazon --lr=1e-5 --epoch=3 --batch_size=64 
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/amazon_us_reviews_Video_v1_00 --bert_output_dir=bert-checkpoint-amazon
INFO:__main__:[Accuracy] : 0.9659,  8917/ 9232
INFO:__main__:[Elapsed Time] : 131532.05513954163ms, 14.232453539549446ms on average

# apply `bert-checkpoint-amazon` to sst2 data
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./bert-checkpoint-amazon
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./bert-checkpoint-amazon --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9336,  1700/ 1821
INFO:__main__:[Elapsed Time] : 28667.39320755005ms, 15.675509106981885ms on average

** fine-tune bert-large-uncased using amazon_us_reviews
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/amazon_us_reviews_Video_v1_00 --bert_output_dir=bert-checkpoint-amazon
INFO:__main__:[Accuracy] : 0.9719,  8973/ 9232
INFO:__main__:[Elapsed Time] : 232065.03987312317ms, 25.12068054216575ms on average

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9264,  1687/ 1821
INFO:__main__:[Elapsed Time] : 46093.640089035034ms, 25.243094727233217ms on average
*** --epoch=10  --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9357,  1704/ 1821
INFO:__main__:[Elapsed Time] : 49913.90776634216ms, 27.320906356140807ms on average

** augment train.txt
$ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw --lower --parallel --preserve_label --n_iter=5 --max_ng=3
$ cp -rf data/sst2/augmented.raw data/sst2/augmented.txt
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --augmented --augmented_filename=augmented.txt
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64 --augmented
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9325,  1698/ 1821
INFO:__main__:[Elapsed Time] : 26077.781438827515ms, 14.253564195318537ms on average

```

</p>
</details>


<details><summary><b>emb_class=albert, enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json
* n_ctx size should be less than 512

* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2
$ python train.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2 --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint 

INFO:__main__:[Accuracy] : 0.9204,  1676/ 1821
INFO:__main__:[Elapsed Time] : 29321ms, 16.055494505494504ms on average

** --bert_model_name_or_path=./embeddings/albert-xxlarge-v2 --batch_size=32
INFO:__main__:[Accuracy] : 0.9577,  1744/ 1821
INFO:__main__:[Elapsed Time] : 104769ms, 57.463186813186816ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

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
$ python train.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64

* enc_class=cls

$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large 
$ python train.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64
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
$ python train.py --config=configs/config-bart-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64

* enc_class=cls

$ python preprocess.py --config=configs/config-bart-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large 
$ python train.py --config=configs/config-bart-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bart-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64
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
* share config-bert-*.json
* n_ctx size should be less than 512

* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator
$ python train.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-base-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint 

INFO:__main__:[Accuracy] : 0.9539,  1737/ 1821
INFO:__main__:[Elapsed Time] : 29602ms, 14.98021978021978ms on average

** --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6
INFO:__main__:[Accuracy] : 0.9566,  1742/ 1821
INFO:__main__:[Elapsed Time] : 54157ms, 28.356593406593408ms on average

** --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6 --epoch=15
INFO:__main__:[Accuracy] : 0.9605,  1749/ 1821
INFO:__main__:[Elapsed Time] : 52163ms, 27.286813186813188ms on average

* enc_lass=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.9522,  1734/ 1821
INFO:__main__:[Elapsed Time] : 25956ms, 14.008791208791209ms on average

** --bert_model_name_or_path=./embeddings/electra-large-discriminator --lr=1e-6 --epoch=15
INFO:__main__:[Accuracy] : 0.9643,  1756/ 1821
INFO:__main__:[Elapsed Time] : 47163ms, 25.685714285714287ms on average
```

</p>
</details>

<br>



# Experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

<br>



# Optimization

- [OPTIMIZATION.md](/OPTIMIZATION.md)
  - Dynamic Quantization
  - Conversion to ONNX
  - Inference with ONNXRUNTIME
  - Quantization
  - Hyper-arameter Search

<br>



# Distillation

- [DISTILLATION.md](/DISTILLATION.md)
  - Augmentation
  - Knowledge Distillation
    - ex) BERT-large, RoBERTa-large, ELECTRA-large -> augmentation/distillation -> GloVe, DenseNet-CNN, DenseNet-DSA

<br>



# Fastformers

- [FASTFORMERS.md](/FASTFORMERS.md)
  - Knowledge Distillation
  - Structured Prunning
  - Quantization

<br>



# TorchServe

- archiving and start torch server
```
$ cd torchserve
$ ./archiver.sh -v -v
$ ./start-torchserve.sh -v -v
```

- request 
```
* health check
$ curl http://localhost:16543/ping
{
  "status": "Healthy"
}

* management api
$ curl http://localhost:16544/models
{
  "models": [
    {
      "modelName": "electra",
      "modelUrl": "electra.mar"
    }
  ]
}

* view all inference apis
$ curl -X OPTIONS http://localhost:16543

* view all management apis
$ curl -X OPTIONS http://localhost:16544

* classify
$ curl -X POST http://localhost:16543/predictions/electra --form data='이 영화는 재미가 있다' | jq
{
  "text": "이 영화는 재미가 있다",
  "results": "1"
}
```

<br>



# Citation

```
@misc{iclassifier,
  author = {dsindex},
  title = {iclassifier},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dsindex/iclassifier}},
}
```

<br>



# References

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
- [Deploying huggingface‘s BERT to production with pytorch/serve](https://medium.com/@freidankm_39840/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)
  - [torchserve](https://aws.amazon.com/ko/blogs/korea/announcing-torchserve-an-open-source-model-server-for-pytorch/)
  - [torchserve management](https://pytorch.org/serve/management_api.html#list-models)
  - [torchserve advanced configuration](https://pytorch.org/serve/configuration.html)

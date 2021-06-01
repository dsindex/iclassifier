# Description

**reference pytorch code for intent(sentence) classification.**

- embedding
  - GloVe, BERT, DistilBERT, mDistilBERT, TinyBERT, MiniLM, MobileBERT, SpanBERT, ALBERT, RoBERTa, XLM-RoBERTa, BART, ELECTRA, DeBERTa, BORT, ConvBERT
- encoding
  - GNB
    - Gaussian Naive Bayes(simple biased model)
  - CNN
    - Convolutional Neural Net
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

<br>



# Pretrained models

- glove
  - [download GloVe6B](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embeddings' dir
  ```
  $ mkdir embeddings
  $ ls embeddings
  glove.6B.zip
  $ unzip glove.6B.zip 
  ```

- BERT-like models(huggingface's [transformers](https://github.com/huggingface/transformers.git))

- [SpanBERT](https://github.com/facebookresearch/SpanBERT/blob/master/README.md)
  - pretrained SpanBERT models are compatible with huggingface's BERT modele except `'bert.pooler.dense.weight', 'bert.pooler.dense.bias'`.

<br>



# Snips data

### experiments summary

|                      | Accuracy (%) | GPU / CPU           | ONNX     | Dynamic   | QAT/FX/DiffQ      | Inference   | Inference+Dynamic | Inference+QAT/FX/DiffQ | Inference+ONNX           | Etc            |
| -------------------- | ------------ | ------------------- | -------- |---------- | ----------------- | ----------- | ----------------- | ---------------------- | ------------------------ | -------------- |    
| GloVe, GNB           | 80.43        | 1.2929  / -         | -        | -         |  -                | -           | -                 | -                      | -                        |                |
| GloVe, CNN           | 97.86        | 1.9874  / 4.1068    | 2.2263   | 4.3975    |  3.0899 / - / -   | 1.9398      | 2.9012            | 1.4329 / - / -         | 0.5270  / FAIL           | threads=14     |
| GloVe, Densenet-CNN  | 97.57        | 3.6094  / -         | 3.0717   | -         |  -                | 4.9481      | -                 | -                      | 0.8658  / FAIL           | threads=14     |
| GloVe, Densenet-DSA  | 97.43        | 7.5007  / -         | 4.4936   | -         |  -                | 7.2086      | -                 | -                      | 1.5420  / FAIL           | threads=14     |
| TinyBERT, CLS        | 97.29        | 6.0523  / -         | 5.3673   | 14.0268   |  -                | 5.7238      | 6.1879            | -                      | **1.7821**  / **1.8446** | threads=14     |
| BERT-small, CLS      | 98.00        | 5.9837  / -         | -        | 15.2820   |  -                | 7.4538      | 7.2436            | -                      | 3.5445  / 2.4141         | threads=14     |
| DistilBERT, CLS      | 97.71        | 8.0221  / 34.3049   | 28.6644  | 31.7260   |  FAIL / FAIL / -  | 16.3812     | 11.2421           | FAIL / FAIL / 13.7443  | 6.1573  / 4.6346         | threads=14     |
| SqueezeBERT, CLS     | 97.29        | 18.0796 / -         | -        | 23.8565   |  -                | 20.3999     | 20.0118           | -                      | 11.9890 / FAIL           | threads=14     |
| MiniLM, CLS          | 96.86        | 12.2094 / -         | 17.5638  | 38.2084   |  -                | 16.8337     | 17.7702           | -                      | 5.0394  / 4.3123         | threads=14     |
| MobileBERT, CLS      | 96.43        | 49.9843 / -         | 46.2151  | 84.4232   |  -                | 51.9885     | 51.4533           | -                      | 15.3492 / 12.4416        | threads=14     |
| BERT-base, CNN       | 97.57        | 12.1273 / -         | -        | -         |  -                | 34.7878     | 30.5454           | -                      | -                        | threads=14     |
| BERT-base, CLS       | 97.43        | 12.7714 / -         | 46.4263  | 49.4747   |  -                | 30.7979     | 24.5353           | -                      | 16.9756                  | threads=14     |
| BERT-base, CLS       | 97.00        | 9.2660  / -         | 31.5400  | 33.4623   |  -                | 16.7419     | 13.5703           | -                      | 11.7487                  | del 8,9,19,11, threads=14 |
| BERT-large, CNN      | **98.00**    | 24.277  / -         | -        | -         |  -                | -           | -                 | -                      | -                        |                |
| BERT-large, CLS      | 97.86        | 23.542  / -         | -        | -         |  -                | -           | -                 | -                      | -                        |                |

```
* GPU / CPU : Elapsed time/example(ms), GPU / CPU  [Tesla V100 1 GPU, Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 2 CPU, 14CORES/1CPU, HyperThreading]
* ONNX : --enable_ort
* Dynamic : --enable_dqm
* Inference : --enable_inference
* QAT(Quantization Aware Training)/FX/DiffQ : --enable_qat / --enable_qat_fx / --enable_diffq
* Inference+Dynamic : --enable_inference --enable_dqm
* Inference+QAT/FX/DiffQ : --enable_inference --enable_qat / --enable_inference --enable_qat_fx / --enable_inference --enable_diffq
* Inference+ONNX : --enable_inference --enable_ort / + --quantize_onnx
* default batch size, learning rate, n_ctx(max_seq_length) : 128, 2e-4, 100
* default epoch : 3
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

** --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/squeezebert-uncased
INFO:__main__:[Accuracy] : 0.9729,   681/  700
INFO:__main__:[Elapsed Time] : 12742.885112762451ms, 18.07960045013646ms on average

** --config=configs/config-bert-cls.json --bert_model_name_or_paht=./embeddings/pytorch.uncased_L-4_H-512_A-8
INFO:__main__:[Accuracy] : 0.9800,   686/  700
INFO:__main__:[Elapsed Time] : 4279.639005661011ms, 5.983798800619887ms on average

** --config=configs/config-bert-cls.json --bert_model_name_or_paht=./embeddings/MiniLM-L12-H384-uncased
INFO:__main__:[Accuracy] : 0.9686,   678/  700
INFO:__main__:[Elapsed Time] : 8598.786115646362ms, 12.209460459042004ms on average

** --config=configs/config-bert-cls.json --bert_model_name_or_paht=./embeddings/TinyBERT_General_4L_312D
INFO:__main__:[Accuracy] : 0.9729,   681/  700
INFO:__main__:[Elapsed Time] : 4330.849170684814ms, 6.052388653734723ms on average

** --config=configs/config-bert-cls.json --bert_model_name_or_paht=./embeddings/mobilebert-uncased
INFO:__main__:[Accuracy] : 0.9643,   675/  700
INFO:__main__:[Elapsed Time] : 35100.63934326172ms, 49.98437324818624ms on average

```

</p>
</details>

<br>



# SST-2 data

### experiments summary

- iclassifier

|                                         | Accuracy (%) | GPU/CPU           | Dynamic                  | Etc           |
| --------------------------------------- | ------------ | ----------------- | ------------------------ | ------------- |
| GloVe, GNB                              | 72.27        | 1.2253  / -       |              - / -       |               |
| GloVe, CNN                              | 83.09        | 1.9943  / 4.5757  |              - / 4.8686  | threads=14    |
| ConceptNet, CNN                         | 84.79        | 2.8304  / -       |              - / -       |               |
| ConceptNet, CNN                         | 84.90        | 2.7672  / -       |              - / -       | optuna        |
| GloVe, DenseNet-CNN                     | 86.38        | 3.6203  / -       |                          | threads=14    |
| ConceptNet, DenseNet-CNN                | 87.26        | 3.7052  / -       |                          |               |
| ConceptNet, DenseNet-CNN                | 87.26        | 3.5377  / -       |                          | optuna        |
| GloVe, DenseNet-DSA                     | 85.34        | 6.2450  / -       |                          |               |
| DistilFromBERT, GloVe, CNN              | 86.16        | 1.7900  / -       |                          | augmented, from large             |
| DistilFromBERT, GloVe, DenseNet-CNN     | 88.52        | 3.6788  / -       |                          | augmented, from large             |
| DistilFromBERT, GloVe, DenseNet-DSA     | 88.14        | 8.4647  / -       |                          | augmented, from large             |
| DistilFromBERT, BERT-small, CLS         | 89.29        | 5.9408  / -       |                          | fastformers, from base            |
| DistilFromBERT, BERT-small, CLS         | 91.49        | 5.9114  / -       |                          | fastformers, augmented, n_iter=20, from base  |
| DistilFromBERT, BERT-small, CLS         | 90.33        | 6.0072  / -       |                          | fastformers, augmented, n_iter=20, from large |
| DistilFromBERT, BERT-small, CLS         | 91.16        | 6.2050  / -       |                          | fastformers, augmented, n_iter=10, from base  |
| DistilFromBERT, BERT-small, CLS         | 91.27        | 6.5702  / -       |                          | fastformers, augmented, n_iter=10, from base, meta pseudo labels |
| DistilFromELECTRA, BERT-small, CLS      | 89.73        | 7.5992  / -       |                          | fastformers, from large           |
| DistilFromRoBERTa, GloVe, CNN           | 86.55        | 1.8483  / -       |                          | augmented, from large             |
| DistilFromRoBERTa, GloVe, DenseNet-CNN  | 88.80        | 3.9580  / -       |                          | augmented, from large             |
| DistilFromRoBERTa, GloVe, DenseNet-DSA  | 88.25        | 8.5627  / -       |                          | augmented, from large             |
| DistilFromELECTRA, GloVe, CNN           | 86.55        | 1.7466  / -       |                          | augmented, from large             |
| DistilFromELECTRA, GloVe, DenseNet-CNN  | 89.79        | 3.6406  / -       |                          | augmented, from large             |
| DistilFromELECTRA, GloVe, DenseNet-DSA  | 88.58        | 8.3708  / -       |                          | augmented, from large             |
| DistilFromELECTRA, DistilBERT, CLS      | 93.52        | 7.4879  / -       |                          | augmented, from large             |
| BERT-tiny, CNN                          | 79.08        | 4.8604  / -       |                          |               |
| BERT-tiny, CLS                          | 80.83        | 3.8461  / -       |                          |               |
| BERT-mini, CNN                          | 83.36        | 7.0983  / -       |                          |               |
| BERT-mini, CLS                          | 83.69        | 5.5521  / -       |                          |               |
| BERT-small, CNN                         | 87.53        | 7.2010  / -       |                          |               |
| BERT-small, CLS                         | 88.25        | 5.9604  / -       |                          |               |
| BERT-medium, CNN                        | 88.58        | 11.9082 / -       |                          |               |
| BERT-medium, CLS                        | 89.24        | 9.5857  / -       |                          |               |
| TinyBERT, CNN                           | 88.14        | 6.9060  / -       |              - / -       |               |
| TinyBERT, CLS                           | 88.19        | 5.9246  / -       |              - / -       |               |
| TinyBERT, CLS                           | 89.84        | 5.9194  / -       |              - / -       | epoch=30      |
| DistilBERT, CNN                         | 89.90        | 9.9362  / -       |              - / 35.7070 | threads=14    |
| **DistilBERT, CLS**                     | 91.10        | 8.9719  / -       |              - / 29.4646 | threads=14    |
| DistilBERT, CLS                         | 92.04        | 6.9790  / -       |              - / -       | epoch=30      |
| mDistilBERT, CLS                        | 87.10        | 7.9757  / -       |              - / -       | epoch=30      |
| MiniLM, CNN                             | 91.49        | 13.5255 / -       |              - / -       |               |
| MiniLM, CLS                             | 91.21        | 12.2066 / -       |              - / -       |               |
| MiniLM, CLS                             | 93.25        | 11.5939 / -       |              - / -       | epoch=30      |
| MobileBERT, CLS                         | 91.05        | 55.0898 / -       |              - / -       |               |
| BERT-base, CNN                          | 92.04        | 14.1576 / -       |                          |               |
| BERT-base, CLS                          | 92.42        | 12.7549 / 62.5050 | 66.4545(92.42) / 50.8080 | threads=14    |
| BERT-base, CLS                          | 93.36        | 15.6755 / -       |              - / -       | fintuned using amazon reviews, epoch=10 |
| BERT-base, CLS                          | 93.25        | 14.2535 / -       |              - / -       | augmented, n_iter=20              |
| BERT-base, CLS                          | 92.81        | 15.2709 / -       |              - / -       | fastformers, augmented, n_iter=10 |
| BERT-base, CLS                          | 93.36        | 15.2605 / -       |              - / -       | fastformers, augmented, n_iter=10, meta pseudo lables |
| BERT-base, CNN                          | 90.55        | 10.6824 / -       |                          | del 8,9,10,11 |
| BERT-base, CLS                          | 91.49        | 8.7747  / 42.8989 | 44.7676(90.61) / 34.3131 | del 8,9,10,11, threads=14         |
| BERT-base, CLS                          | 90.23        | 7.0241  / -       |                          | del 6,7,8,9,10,11, threads=14     |
| BERT-base, CLS                          | 86.66        | 5.8868  / -       |                          | del 4,5,6,7,8,9,10,11, threads=14 |
| BERT-large, CNN                         | 93.08        | 28.6490 / -       |                          |               |
| BERT-large, CLS                         | 94.12        | 22.3767 / -       |                          |               |
| BERT-large, CLS                         | 93.57        | 27.3209 / -       |                          | fintuned using amazon reviews     |
| BERT-large, CNN                         | 88.47        | 14.7813 / -       |                          | del 12~23     |
| BERT-large, CLS                         | 86.71        | 12.1560 / -       |                          | del 12~23     |
| SqueezeBERT, CNN                        | 90.61        | 19.2879 / -       |                          | epoch=20      |
| SqueezeBERT, CLS                        | 90.12        | 17.4998 / -       |                          | epoch=20      |
| SpanBERT-base, CNN                      | 91.82        | 15.2098 / -       |                          |               |
| SpanBERT-base, CLS                      | 91.49        | 13.1516 / -       |                          |               |
| SpanBERT-large, CNN                     | 93.90        | 26.8609 / -       |                          |               |
| SpanBERT-large, CLS                     | 93.96        | 26.0445 / -       |                          |               |
| ALBERT-base, CNN                        | 92.04        | 16.0554 / -       |                          | epoch=10      |
| ALBERT-base, CLS                        | 90.01        | 14.6725 / -       |                          | epoch=10      |
| ALBERT-xxlarge, CNN                     | 95.77        | 57.4631 / -       |                          | epoch=10      |
| ALBERT-xxlarge, CLS                     | 94.45        | 51.8027 / -       |                          | epoch=10      |
| RoBERTa-base, CNN                       | 92.92        | 15.1016 / -       |                          | epoch=10      |
| RoBERTa-base, CLS                       | 93.03        | 14.6736 / -       |                          | epoch=10      |
| XLM-RoBERTa-base, CLS                   | 91.49        | 14.2246 / -       |                          | epoch=10      |
| RoBERTa-base, CNN                       | 92.26        | 11.5241 / -       |                          | del 8,9,10,11 , epoch=10 |
| RoBERTa-base, CLS                       | 91.76        | 10.0296 / -       |                          | del 8,9,10,11 , epoch=10 |
| RoBERTa-large, CNN                      | 95.55        | 26.9807 / -       |                          | epoch=10      |
| RoBERTa-large, CLS                      | 95.66        | 23.7395 / -       |                          | epoch=10      |
| XLM-RoBERTa-large, CLS                  | 92.04        | 24.8045 / -       |                          | epoch=10      |
| BART-large, CNN                         | 94.45        | 35.1708 / -       |                          | epoch=10      |
| BART-large, CLS                         | 94.89        | 33.3862 / -       |                          | epoch=10      |
| ELECTRA-base, CNN                       | 95.39        | 14.9802 / -       |                          | epoch=10      |
| ELECTRA-base, CLS                       | 95.22        | 14.0087 / -       |                          | epoch=10      |
| ELECTRA-large, CNN                      | 96.05        | 27.2868 / -       |                          | epoch=15      |
| ELECTRA-large, CLS                      | **96.43**    | 25.6857 / -       |                          | epoch=15      |
| DeBERTa-base, CLS                       | 93.41        | 26.1533 / -       |                          | epoch=10      |
| DeBERTa-large, CLS                      | 94.95        | 62.4272 / -       |                          | epoch=10      |
| DeBERTa-v2-xlarge, CLS                  | 96.21        | 54.1388 / -       |                          | epoch=10      |
| BORT, CLS                               | 77.98        | 6.1346  / -       |                          | epoch=10      |
| ConvBERT, CLS                           | 77.48        | 22.6815 / -       |                          | epoch=10      |
| GPT2-large, CLS                         | 94.45        | 36.5779 / -       |                          | epoch=10      |
| GPT2-xlarge, CLS                        | -            | -       / -       |                          | epoch=10      |

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

INFO:__main__:[Accuracy] : 0.8309,  1513/ 1821
INFO:__main__:[Elapsed Time] : 3725.560188293457ms, 1.994345607338371ms on average

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

** --bert_model_name_or_path=embeddings/TinyBERT_General_4L_312D
INFO:__main__:[Accuracy] : 0.8814,  1605/ 1821
INFO:__main__:[Elapsed Time] : 12672.252178192139ms, 6.906037802224631ms on average

** --configs/config-distilbert-cnn.json --bert_model_name_or_path=embeddings/distilbert-base-uncased
INFO:__main__:[Accuracy] : 0.8990,  1637/ 1821
INFO:__main__:[Elapsed Time] : 18193ms, 9.936263736263736ms on average

** --configs/config-bert-cnn.json --bert_model_name_or_path=embeddings/MiniLM-L12-H384-uncased
INFO:__main__:[Accuracy] : 0.9149,  1666/ 1821
INFO:__main__:[Elapsed Time] : 24706.911087036133ms, 13.525558042002249ms on average

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

** --bert_model_name_or_path=embeddings/TinyBERT_General_4L_312D
INFO:__main__:[Accuracy] : 0.8819,  1606/ 1821
INFO:__main__:[Elapsed Time] : 10878.073692321777ms, 5.92467928980733ms on average

** --bert_model_name_or_path=embeddings/TinyBERT_General_4L_312D --epoch=30
INFO:__main__:[Accuracy] : 0.8984,  1636/ 1821
INFO:__main__:[Elapsed Time] : 10882.532119750977ms, 5.9194347360631925ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=embeddings/distilbert-base-uncased
INFO:__main__:[Accuracy] : 0.9110,  1659/ 1821
INFO:__main__:[Elapsed Time] : 16431ms, 8.971978021978021ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=embeddings/distilbert-base-uncased --epoch=30
INFO:__main__:[Accuracy] : 0.9204,  1676/ 1821
INFO:__main__:[Elapsed Time] : 12806.593418121338ms, 6.979021790263417ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=embeddings/distilbert-base-multilingual-cased --epoch=30
INFO:__main__:[Accuracy] : 0.8710,  1586/ 1821
INFO:__main__:[Elapsed Time] : 14596.29511833191ms, 7.975767077980461ms on average

** --configs/config-bert-cls.json --bert_model_name_or_path=embeddings/MiniLM-L12-H384-uncased
INFO:__main__:[Accuracy] : 0.9121,  1661/ 1821
INFO:__main__:[Elapsed Time] : 22304.69584465027ms, 12.206664845183655ms on average

** --configs/config-bert-cls.json --bert_model_name_or_path=embeddings/MiniLM-L12-H384-uncased --epoch=30
INFO:__main__:[Accuracy] : 0.9325,  1698/ 1821
INFO:__main__:[Elapsed Time] : 21175.854682922363ms, 11.593997216486668ms on average

** --configs/config-bert-cls.json --bert_model_name_or_path=embeddings/mobilebert-uncased
INFO:__main__:[Accuracy] : 0.9105,  1658/ 1821
INFO:__main__:[Elapsed Time] : 100408.40005874634ms, 55.08984157017299ms on average

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

** --bert_model_name_or_path=./embeddings/xlm-roberta-base
INFO:__main__:[Accuracy] : 0.9149,  1666/ 1821
INFO:__main__:[Elapsed Time] : 25997.645378112793ms, 14.224677033476777ms on average

** --bert_model_name_or_path=./embeddings/xlm-roberta-large
INFO:__main__:[Accuracy] : 0.9204,  1676/ 1821
INFO:__main__:[Elapsed Time] : 45271.38304710388ms, 24.804521654988385ms on average

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


<details><summary><b>emb_class=deberta, enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json
* n_ctx size should be less than 512

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/deberta-base
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/deberta-base --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64
```

- evaluation
```

* enc_lass=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9341,  1701/ 1821
INFO:__main__:[Elapsed Time] : 47751.57833099365ms, 26.153329571524818ms on average

** --bert_model_name_or_path=./embeddings/deberta-large --batch_size=32 --gradient_accumulation_steps=2
INFO:__main__:[Accuracy] : 0.9495,  1729/ 1821
INFO:__main__:[Elapsed Time] : 113818.21751594543ms, 62.427293075310004ms on average

** --bert_model_name_or_path=./embeddings/deberta-v2-xlarge --batch_size=16 --gradient_accumulation_steps=4
INFO:__main__:[Accuracy] : 0.9621,  1752/ 1821
INFO:__main__:[Elapsed Time] : 98728.88159751892ms, 54.138861121712154ms on average

```

</p>
</details>


<details><summary><b>emb_class=bort, enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json
* n_ctx size should be less than 512

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bort

$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bort --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64

```

- evaluation
```

* enc_lass=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.7798,  1420/ 1821
INFO:__main__:[Elapsed Time] : 11250.777959823608ms, 6.134679160275302ms on average

```

</p>
</details>


<details><summary><b>emb_class=convbert, enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json
* n_ctx size should be less than 512

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/conv-bert-medium-small

$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/conv-bert-medium-small --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64 

```

- evaluation
```

* enc_lass=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.7748,  1411/ 1821
INFO:__main__:[Elapsed Time] : 41405.57098388672ms, 22.681565337128692ms on average

```

</p>
</details>


<details><summary><b>emb_class=gpt, enc_class=cnn | cls</b></summary>
<p>

- train
```
* n_ctx size should be less than 512

* enc_class=cls

$ python preprocess.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/gpt2-xl
$ python train.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/gpt2-xl --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=8 --gradient_accumulation_steps=4 --eval_and_save_steps=32 --zero_stage=3

# 1 node, 2 gpu
$ export NCCL_DEBUG=INFO
$ python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 --use_env --node_rank 0 --master_addr 127.0.0.1 --master_port 24158 train.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/gpt2-xl --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=8 --gradient_accumulation_steps=4 --eval_and_save_steps=32 --zero_stage=2

!! fail to train, may need increasing the number of gpus !!

** gpt2-large
$ python train.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/gpt2-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=16 --gradient_accumulation_steps=2


** gpt2-large, torch.distributed.lanuch

$ export NCCL_DEBUG=INFO
$ python -m torch.distributed.launch --nnodes 1 --nproc_per_node 2 --use_env --node_rank 0 --master_addr 127.0.0.1 --master_port 24158 train.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/gpt2-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=8 --gradient_accumulation_steps=4 --eval_and_save_steps=32 --zero_stage=2


** gpt2-large, accelerate launch
$ python -m pip install deepspeed, apex
# fixing error
$ vi /usr/local/lib/python3.6/dist-packages/accelerate/deepspeed_utils.py
if is_apex_available():
    #import amp
    from apex import amp
$ accelerate config
In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU): 2
How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
Do you want to use DeepSpeed? [yes/NO]: yes
What should be your DeepSpeed's ZeRO optimization stage (0, 1, 2, 3)? [2]: 2
Where to offload optimizer states? [NONE/cpu/nvme]: cpu
How many gradient accumulation steps you're passing in your script? [1]: 4
How many processes in total will you use? [1]: 2
Do you wish to use FP16 (mixed precision)? [yes/NO]: NO
$ accelerate launch train.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/gpt2-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=8 --gradient_accumulation_steps=4 --eval_and_save_steps=32

```

- evaluation
```

* enc_lass=cls

$ python evaluate.py --config=configs/config-gpt-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

** --bert_model_name_or_path=./embeddings/gpt2-large
INFO:__main__:[Accuracy] : 0.9445,  1720/ 1821
INFO:__main__:[Elapsed Time] : 66759.70959663391ms, 36.57794352416154ms on average


```

</p>
</details>


<br>



# Experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

<br>



# Optimization

- [OPTIMIZATION.md](/OPTIMIZATION.md)
  - Quantization
    - Dynamic
    - Quantization Aware Training
  - ONNX/ONNXRUNTIME, ONNX Quantization
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
    - Meta Pseudo Labels
  - Structured Prunning
  - ONNX Quantization

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

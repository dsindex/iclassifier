# 한국어 데이터 대상 실험

### Data

- [NSMC(Naver Sentiment Movie Corpus)](https://github.com/e9t/nsmc)
  - setup
    - './data/clova_sentiments/'
      - 'train.txt', 'valid.txt', 'test.txt'
      - 'test.txt'는 제공하지 않으므로 'valid.txt'를 복사해서 사용.
      - '*.txt' 데이터의 포맷은 './data/snips'의 데이터 포맷과 동일.
    - './data/clova_sentiments_morph'
      - `형태소분석기 tokenizer`를 적용한 데이터.
      - 'train.txt', 'valid.txt', 'test.txt'.
      - 형태소분석기는 [khaiii](https://github.com/kakao/khaiii) 등 사용 가능.

- [korean-hate-speech](https://github.com/kocohub/korean-hate-speech)
  - setup
    - './data/korean_hate_speech/'
      - 'train.txt', 'valid.txt', 'test.txt'
      - 'test.txt'는 제공하지 않으므로 'valid.txt'를 복사해서 사용.
      - 원문의 'comment', 'hate' label만 사용
      - data augmentation(distillation) 등에 활용하기 위해서 'unlabeled' 데이터도 복사.
        - 데이터 사이즈가 제법 크기 때문에, git에 추가하지 않고, 다운받아서 사용.
    - './data/korean_hate_speech_morph'
      - `형태소분석기 tokenizer`를 적용한 데이터.
    - './data/korean_bias_speech/'
      - 원문의 'comment', 'bias' label만 사용
    - './data/korean_bias_speech_morph'
      - `형태소분석기 tokenizer`를 적용한 데이터.

### Pretrained models

##### GloVe

- [Standford GloVe code](https://github.com/stanfordnlp/GloVe)를 이용해서 학습.
  - 한국어 문서 데이터 준비.
    - 다양한 문서 데이터(위키, 백과, 뉴스, 블로그 등등)를 크롤링.
  - 형태소분석기 tokenizer를 적용해서 형태소 단위로 변경한 데이터를 이용해서 학습 진행.
  - ex) kor.glove.300k.300d.txt (inhouse)

##### BERT

- [google original tf code](https://github.com/google-research/bert)를 이용해서 학습.
  - 한국어 문서 데이터 준비.
    - 위 한국어 GloVe 학습에 사용한 데이터를 그대로 이용.
  - `character-level bpe`
    - vocab.txt는 [sentencepiece](https://github.com/google/sentencepiece)를 이용해서 생성.
    - ex) pytorch.all.bpe.4.8m_step, pytorch.large.all.whitespace_bpe.7m_step (inhouse)
  - `character-level bpe + 형태소분석기`
    - ex) pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step, pytorch.large.all.dha_s2.9.4_d2.9.27_bpe.7m_step (inhouse)
  - `형태소분석기`
    - ex) pytorch.all.dha.2.5m_step (inhouse), pytorch.all.dha_s2.9.4_d2.9.27.10m_step (inhouse)

##### DistilBERT

- [training-distilbert](https://github.com/dsindex/transformers_examples#training-distilbert)
  - ex) `kor-distil-bpe-bert.v1` (inhouse)

##### ELECTRA

- monologg koelectra-base
  - [koelectra-base-discriminator](https://huggingface.co/monologg/koelectra-base-discriminator)

- [electra](https://github.com/dsindex/electra#pretraining-electra)를 이용해서 학습.
  - 한국어 문서 데이터 준비.
    - 위 한국어 GloVe 학습에 사용한 데이터를 그대로 이용.
  - [README.md](https://github.com/dsindex/electra/blob/master/README.md)
  - [train.sh](https://github.com/dsindex/electra/blob/master/train.sh)
    - ex) `kor-electra-base-bpe-30k-512-1m` (inhouse)

### NMSC data

- iclassifier

|                                     | Accuracy (%) | GPU / CPU         | CONDA   | Etc        |
| ----------------------------------- | ------------ | ----------------- | ------- | ---------- |
| GloVe, CNN                          | 87.31        | 1.9479  / 3.5353  |         | threads=14 |
| **GloVe, DenseNet-CNN**             | 88.18        | 3.4614  / 8.3434  |         | threads=14 |
| DistilFromBERT, GloVe, DenseNet-CNN | 89.21        | 3.5383  / -       |         |            |
| GloVe, DenseNet-DSA                 | 87.66        | 6.9731  / -       |         |            |
| bpe BERT(4.8m), CNN                 | 90.11        | 16.5453 / -       |         |            |
| bpe BERT(4.8m), CLS                 | 89.91        | 14.9586 / -       |         |            |
| bpe BERT(4.8m), CNN                 | 88.62        | 10.7023 / 73.4141 |         | del 8,9,10,11, threads=14 |
| bpe BERT(4.8m), CLS                 | 88.92        | 9.3280  / 70.3232 |         | del 8,9,10,11, threads=14 |
| bpe DistilBERT(4.8m), CNN           | 88.39        | 9.6396  / -       | 38.7144 |        , threads=14       |
| bpe DistilBERT(4.8m), CLS           | 88.55        | 8.2834  / -       | 31.5655 |        , threads=14       |
| bpe BERT-large, CNN                 | -            | -       / -       |         |            |
| bpe BERT-large, CLS                 | -            | -       / -       |         |            |
| dha BERT(2.5m), CNN                 | **90.25**    | 15.5738 / -       |         |            |
| dha BERT(2.5m), CLS                 | 90.18        | 13.3390 / -       |         |            |
| dha BERT(2.5m), CNN                 | 88.88        | 10.5157 / 72.7777 |         | del 8,9,10,11, threads=14         |
| dha BERT(2.5m), CLS                 | 88.81        | 8.9836  / 68.4545 | 50.7474 | del 8,9,10,11, threads=14         |
| dha BERT(2.5m), CLS                 | 88.29        | 7.2027  / 53.6363 | 38.3333 | del 6,7,8,9,10,11, threads=14     |
| dha BERT(2.5m), CLS                 | 87.54        | 5.7645  / 36.8686 | 28.2626 | del 4,5,6,7,8,9,10,11, threads=14 |
| dha-bpe BERT(4m), CNN               | 89.07        | 14.9454 / -       |         |            |
| dha-bpe BERT(4m), CLS               | 89.01        | 12.7981 / -       |         |            |
| dha-bpe BERT-large, CNN             | -            | -       / -       |         |            |
| dha-bpe BERT-large, CLS             | -            | -       / -       |         |            |
| dha BERT(10m), CNN                  | 89.08        | 15.3276 / -       |         |            |
| dha BERT(10m), CLS                  | 89.25        | 12.7876 / -       |         |            |
| KoELECTRA-Base, CNN                 | 89.51        | 15.5452 / -       |         |            |
| KoELECTRA-Base, CLS                 | 89.63        | 14.2667 / -       |         |            |
| bpe ELECTRA-base(30k-512-1m) , CNN  | 88.07        | 16.2737 / -       |         |            |
| bpe ELECTRA-base(30k-512-1m) , CLS  | 88.26        | 14.2356 / -       |         |            |

```
* GPU/CPU : Elapsed time/example(ms), GPU / CPU(pip 1.2.0)
* CONDA : conda pytorch=1.2.0 / conda pytorch=1.5.0
* default batch size, learning rate, n_ctx(max_seq_length) : 128, 2e-4, 100
```

- [HanBert-nsmc](https://github.com/monologg/HanBert-nsmc#results), [KoELECTRA](https://github.com/monologg/KoELECTRA)

|                   | Accuracy (%) | Etc        |
| ----------------- | ------------ | ---------- |
| KoELECTRA-Base    | 90.21        |            |
| KoELECTRA-Base-v2 | 89.70        | vocab from https://github.com/enlipleai/kor_pretrain_LM |
| XML-RoBERTa       | 89.49        |            |
| HanBert-54kN      | 90.16        |            |
| HanBert-54kN-IP   | 88.72        |            |
| KoBERT            | 89.63        |            |
| DistilKoBERT      | 88.41        |            |
| Bert-Multilingual | 87.07        |            |
| FastText          | 85.50        |            |

- [KoBERT](https://github.com/SKTBrain/KoBERT#naver-sentiment-analysis)

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT            | 90.1         |

- [aisolab/nlp_classification](https://github.com/aisolab/nlp_classification)
  - 비교를 위해서, 여기에서는 데이터를 동일하게 맞추고 재실험.
  - `--epoch=10 --learning_rate=5e-4 --batch_size=128`

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT(STKBERT)   | 89.35        |
| ETRIBERT          | 89.99        |


### GloVe

<details><summary><b>enc_class=cnn</b></summary>
<p>

- train
```
$ python preprocess.py --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --data_dir=data/clova_sentiments_morph --lr_decay_rate=0.9 --embedding_trainable
```

- evaluation
```
$ python evaluate.py --data_dir=./data/clova_sentiments_morph 

INFO:__main__:[Accuracy] : 0.8731, 43653/49997
INFO:__main__:[Elapsed Time] : 97481ms, 1.9479358348667895ms on average
```

</p>
</details>


<details><summary><b>enc_class=densenet-cnn</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --lr_decay_rate=0.9

* iee_corpus_morph
$ python preprocess.py --config=configs/config-densenet-cnn-iee.json --data_dir=data/iee_corpus_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn-iee.json --data_dir=data/iee_corpus_morph --lr_decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/clova_sentiments_morph 

INFO:__main__:[Accuracy] : 0.8818, 44087/49997
INFO:__main__:[Elapsed Time] : 173152ms, 3.4614969197535803ms on average

INFO:__main__:[Accuracy] : 0.8799, 43991/49997
INFO:__main__:[Elapsed Time] : 179413.97333145142ms, 3.58712682724762ms on average

* --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8804, 44017/49997
INFO:__main__:[Elapsed Time] : 160672.52135276794ms, 3.211939556139528ms on average

INFO:__main__:[Accuracy] : 0.8803, 44012/49997
INFO:__main__:[Elapsed Time] : 161198.18258285522ms, 3.2225625831057085ms on average

* iee_corpus_morph
$ python evaluate.py --config=configs/config-densenet-cnn-iee.json --data_dir=./data/iee_corpus_morph 

```

</p>
</details>


<details><summary><b>enc_class=densenet-dsa</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph --lr_decay_rate=0.9

* iee_corpus_morph
$ python preprocess.py --config=configs/config-densenet-dsa-iee.json --data_dir=data/iee_corpus_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-dsa-iee.json --data_dir=data/iee_corpus_morph --lr_decay_rate=0.9 --batch_size=256
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=./data/clova_sentiments_morph

INFO:__main__:[Accuracy] : 0.8766, 43827/49997
INFO:__main__:[Elapsed Time] : 348722ms, 6.973197855828467ms on average

INFO:__main__:[Accuracy] : 0.8744, 43715/49997
INFO:__main__:[Elapsed Time] : 522451ms, 10.447275782062565ms on average

INFO:__main__:[Accuracy] : 0.8743, 43710/49997
INFO:__main__:[Elapsed Time] : 404403.4924507141ms, 8.087184512335755ms on average

* softmax masking
INFO:__main__:[Accuracy] : 0.8747, 43732/49997
INFO:__main__:[Elapsed Time] : 596904ms, 11.936694935594847ms on average

* --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8735, 43670/49997
INFO:__main__:[Elapsed Time] : 410616.1913871765ms, 8.211271306382855ms on average

* iee_corpus_morph
$ python evaluate.py --config=configs/config-densenet-dsa-iee.json --data_dir=./data/iee_corpus_morph
```

</p>
</details>


### BERT(pytorch.all.bpe.4.8m_step, pytorch.large.all.whitespace_bpe.7m_step)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments/

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments/
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8945, 44723/49997
INFO:__main__:[Elapsed Time] : 734983ms, 14.697815825266021ms on average

** --bert_remove_layers=8,9,10,11
INFO:__main__:[Accuracy] : 0.8862, 44309/49997
INFO:__main__:[Elapsed Time] : 535186ms, 10.702336186894952ms on average

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9011, 45053/49997
INFO:__main__:[Elapsed Time] : 827306ms, 16.545303624289943ms on average

** --configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8839, 44190/49997
INFO:__main__:[Elapsed Time] : 482054.96978759766ms, 9.639614557722052ms on average

** --bert_model_name_or_path=./embeddings/pytorch.large.all.whitespace_bpe.7m_step --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30


* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8931, 44653/49997
INFO:__main__:[Elapsed Time] : 672027ms, 13.439275142011361ms on average

INFO:__main__:[Accuracy] : 0.8959, 44790/49997
INFO:__main__:[Elapsed Time] : 703563ms, 14.07036562925034ms on average

** --bert_remove_layers=8,9,10,11
INFO:__main__:[Accuracy] : 0.8892, 44457/49997
INFO:__main__:[Elapsed Time] : 466825ms, 9.32800624049924ms on average

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8991, 44952/49997
INFO:__main__:[Elapsed Time] : 747975ms, 14.958656692535403ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8855, 44271/49997
INFO:__main__:[Elapsed Time] : 414233.4134578705ms, 8.283499222067283ms on average

** --bert_model_name_or_path=./embeddings/pytorch.large.all.whitespace_bpe.7m_step --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30


```

</p>
</details>


### BERT(pytorch.all.dha.2.5m_step)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph/

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=3 --batch_size=64 --data_dir=./data/clova_sentiments_morph/
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=./data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8996, 44976/49997
INFO:__main__:[Elapsed Time] : 743973ms, 14.877990239219137ms on average

** --bert_remove_layers=8,9,10,11
INFO:__main__:[Accuracy] : 0.8888, 44438/49997
INFO:__main__:[Elapsed Time] : 525917ms, 10.515781262501ms on average

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=20
INFO:__main__:[Accuracy] : 0.9025, 45123/49997
INFO:__main__:[Elapsed Time] : 778762ms, 15.573805904472358ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=./data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8941, 44701/49997
INFO:__main__:[Elapsed Time] : 718417ms, 14.36640931274502ms on average

** --bert_remove_layers=8,9,10,11
INFO:__main__:[Accuracy] : 0.8881, 44401/49997
INFO:__main__:[Elapsed Time] : 449263ms, 8.983638691095287ms on average

** --bert_remove_layers=6,7,8,9,10,11
INFO:__main__:[Accuracy] : 0.8829, 44143/49997
INFO:__main__:[Elapsed Time] : 360213ms, 7.202776222097768ms on average

** --bert_remove_layers=4,5,6,7,8,9,10,11
INFO:__main__:[Accuracy] : 0.8754, 43765/49997
INFO:__main__:[Elapsed Time] : 288307ms, 5.764541163293064ms on average

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9018, 45089/49997
INFO:__main__:[Elapsed Time] : 666997.1199035645ms, 13.339050636929372ms on average

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=5e-5
INFO:__main__:[Accuracy] : 0.8967, 44831/49997
INFO:__main__:[Elapsed Time] : 705669ms, 14.11208896711737ms on average


```

</p>
</details>


### BERT(pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step, pytorch.large.all.dha_s2.9.4_d2.9.27_bpe.7m_step)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph

* enc_class=cls

$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8907, 44533/49997
INFO:__main__:[Elapsed Time] : 747351ms, 14.945475638051045ms on average

** --bert_model_name_or_path=./embeddings/pytorch.large.all.dha_s2.9.4_d2.9.27_bpe.7m_step --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30


* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8901, 44503/49997
INFO:__main__:[Elapsed Time] : 639988ms, 12.798163853108248ms on average

** --bert_model_name_or_path=./embeddings/pytorch.large.all.dha_s2.9.4_d2.9.27_bpe.7m_step --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30


```

</p>
</details>


### BERT(pytorch.all.dha_s2.9.4_d2.9.27.10m_step)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph/

* enc_class=cls

$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha_s2.9.4_d2.9.27.10m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=3 --batch_size=64 --data_dir=./data/clova_sentiments_morph/
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=./data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8908, 44536/49997
INFO:__main__:[Elapsed Time] : 766457ms, 15.327646211696935ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=./data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8925, 44622/49997
INFO:__main__:[Elapsed Time] : 639463ms, 12.787603008240659ms on average
```

</p>
</details>


### ELECTRA(koelectra-base-discriminator)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-electra-cnn.json --bert_model_name_or_path=./embeddings/koelectra-base-discriminator --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-electra-cnn.json --bert_model_name_or_path=./embeddings/koelectra-base-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64 --data_dir=./data/clova_sentiments

* enc_class=cls

$ python preprocess.py --config=configs/config-electra-cls.json --bert_model_name_or_path=./embeddings/koelectra-base-discriminator --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-electra-cls.json --bert_model_name_or_path=./embeddings/koelectra-base-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64 --data_dir=./data/clova_sentiments
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-electra-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8937, 44684/49997
INFO:__main__:[Elapsed Time] : 784375ms, 15.636230898471878ms on average

** --use_transformers_optimizer --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8951, 44750/49997
INFO:__main__:[Elapsed Time] : 777338ms, 15.54522361788943ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8930, 44646/49997
INFO:__main__:[Elapsed Time] : 721693ms, 14.425894071525722ms on average

** --use_transformers_optimizer --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8963, 44814/49997
INFO:__main__:[Elapsed Time] : 713403ms, 14.266721337707017ms on average

```

</p>
</details>

### ELECTRA(kor-electra-base-bpe-30k-512-1m)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-electra-cnn.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-30k-512-1m --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-electra-cnn.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-30k-512-1m --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=10 --batch_size=64 --data_dir=./data/clova_sentiments 

* enc_class=cls

$ python preprocess.py --config=configs/config-electra-cls.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-30k-512-1m --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-electra-cls.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-30k-512-1m --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=10 --batch_size=64 --data_dir=./data/clova_sentiments 
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-electra-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=1e-5 --batch_size=64
INFO:__main__:[Accuracy] : 0.8807, 44034/49997
INFO:__main__:[Elapsed Time] : 813746.9305992126ms, 16.273748160305097ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

** --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=1e-5 --batch_size=64
INFO:__main__:[Accuracy] : 0.8826, 44126/49997
INFO:__main__:[Elapsed Time] : 711834.1734409332ms, 14.23564201088388ms on average

```

</p>
</details>


### korean-hate-speech data

- iclassifier

|                                     | Bias Accuracy (%) | Hate Accuracy (%) | GPU / CPU         | CONDA   | Etc                  |
| ----------------------------------- | ----------------- | ----------------- | ----------------- | ------- | -------------------- |
| GloVe, DenseNet-CNN                 | 72.61             | 61.78             | 3.7602  / -       |         |                      |
| DistilFromBERT, GloVe, DenseNet-CNN | -                 | 64.97             | 3.8358  / -       |         |                      |
| DistilFromBERT, GloVe, DenseNet-CNN | -                 | -                 | -       / -       |         | unlabeled data used  |
| dha BERT(2.5m), CNN                 | 83.44             | 67.09             | 15.8797 / -       |         |                      |
| dha BERT(2.5m), CLS                 | 82.80             | 64.76             | 12.8167 / -       |         |                      |

```
* GPU/CPU : Elapsed time/example(ms), GPU / CPU(pip 1.2.0)
* CONDA : conda pytorch=1.2.0 / conda pytorch=1.5.0
* default batch size, learning rate, n_ctx(max_seq_length) : 128, 2e-4, 100
```

- [korean-hate-speech-koelectra](https://github.com/monologg/korean-hate-speech-koelectra)

|                   | Bias Accuracy (%) | Hate Accuracy (%) | Etc                                  |
| ----------------- | ----------------- | ----------------- | ------------------------------------ |
| KoELECTRA-base    | 82.28             | 67.25             | with title, bias/hate joint training |


### GloVe

<details><summary><b>enc_class=densenet-cnn</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/korean_hate_speech_morph --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-cnn.pt

```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-cnn.pt
INFO:__main__:[Accuracy] : 0.6178,   291/  471
INFO:__main__:[Elapsed Time] : 1848.5705852508545ms, 3.760207967555269ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.7261,   342/  471
INFO:__main__:[Elapsed Time] : 2106.7402362823486ms, 4.309217473293873ms on average

```

</p>
</details>


### BERT(pytorch.all.dha.2.5m_step)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/korean_hate_speech_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech_morph --save_path=pytorch-model-kor-bert.pt

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/korean_hate_speech_morph
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech_morph --save_path=pytorch-model-kor-bert.pt
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/korean_hate_speech_morph --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
INFO:__main__:[Accuracy] : 0.6709,   316/  471
INFO:__main__:[Elapsed Time] : 7566.187143325806ms, 15.879765469977196ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8344,   393/  471
INFO:__main__:[Elapsed Time] : 8157.46545791626ms, 17.114992344633063ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/korean_hate_speech_morph --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
INFO:__main__:[Accuracy] : 0.6476,   305/  471
INFO:__main__:[Elapsed Time] : 6128.332614898682ms, 12.81673198050641ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8280,   390/  471
INFO:__main__:[Elapsed Time] : 6393.482446670532ms, 13.385195427752556ms on average

```

</p>
</details>



# 한국어 데이터 대상 실험

### Data

- [Naver sentiment movie corpus](https://github.com/e9t/nsmc)
  - setup
    - 데이터를 다운받아서 './data/clova_sentiments/' 디렉토리 아래
      - 'train.txt', 'valid.txt', 'test.txt' 생성.
      - 'test.txt'는 제공하지 않으므로 'valid.txt'를 복사해서 사용.
      - '*.txt' 데이터의 포맷은 './data/snips'의 데이터 포맷과 동일.
    - `형태소분석기 tokenizer`를 적용한 데이터도 './data/clova_sentiments_morph' 디렉토리 아래 생성.
      - 'train.txt', 'valid.txt', 'test.txt'.
      - 형태소분석기는 [khaiii](https://github.com/kakao/khaiii) 등 사용 가능.
  - previous result
    - [SKT Brain에서 공개한 KoBERT를 적용한 성능](https://github.com/SKTBrain/KoBERT#naver-sentiment-analysis)
      - valid acc : **90.1%**

### GLOVE model

- 한국어 문서 데이터 준비
  - 다양한 문서 데이터를 크롤링

- [Standford GloVe code](https://github.com/stanfordnlp/GloVe)를 이용해서 한국어 GLOVE 학습
  - ex) kor.glove.300k.300d.txt

### BERT model

- 한국어 문서 데이터 준비
  - 위 한국어 GLOVE 학습에 사용한 데이터를 그대로 이용

- [google original tf code](https://github.com/google-research/bert)를 이용해서 학습
  - [sentencepiece](https://github.com/google/sentencepiece) character-level bpe tokenizer
    - ex) all.bpe.4.8m_step
  - `character-level bpe + 형태소분석기`
    - ex) all.dha_s2.9.4_d2.9.27_bpe.4m_step
  - `형태소분석기`
    - ex) all.dha.2.5m_step, all.dha_s2.9.4_d2.9.27.10m_step

- huggingface 포맷으로 변환
  - [convert_bert_orignal_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/master/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py) 스크립트를 이용해서 변환.
  ```
  $ python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=all.bpe.4.8m_step/model.ckpt-4780000 --bert_config_file=all.bpe.4.8m_step/bert_config.json --pytorch_dump_path=pytorch_model.bin
  * 나머지 필요한 파일들은 huggingface에서 배포된 bert-base-cased에 있는 파일들을 복사해서 사용.
  * vocab.txt는 tf에 있는 것을 그대로 활용.
  * config.json의 vocab_size 설정 필요.
  ```

### RoBERTa model

- 한국어 문서 데이터 준비
  - 위 한국어 GLOVE 학습에 사용한 데이터를 그대로 이용

- [transformers](https://github.com/huggingface/transformers)
  - [train-kor-roberta.sh](https://github.com/dsindex/transformers_examples/blob/master/train-kor-roberta.sh)
  - [tokenizers](https://github.com/dsindex/transformers_examples/blob/master/train-tokenizer.py) byte-level bpe tokenizer
  - ex) `kor-roberta-base.v1`
  - 학습 실패
  - fairseq를 이용해서 학습하고 이를 transformers format으로 변환하는 방식으로 진행 필요.

### ELECTRA model

- 위에서 사용한 문서로 새롭게 학습하기 전에, 기존에 huggingface에 올라온 monologg에서 학습시킨 모델을 사용해서 실험.
  - [monologg](https://huggingface.co/monologg/koelectra-base-discriminator)

- 한국어 문서 데이터 준비
  - 위 한국어 GLOVE 학습에 사용한 데이터를 그대로 이용

- [electra](https://github.com/dsindex/electra#pretraining-electra)
  - fork
    - [README.md](https://github.com/dsindex/electra/blob/master/README.md)
    - [train.sh](https://github.com/dsindex/electra/blob/master/train.sh)
    - ex) `kor-electra-base-bpe-128-1m`

### Experiments summary

- iclassifier

|                                 | Accuracy (%) | GPU / CPU         | Etc        |
| ------------------------------- | ------------ | ----------------- | ---------- |
| Glove, CNN                      | 87.31        | 1.9479  / 3.5353  | threads=14 |
| **Glove, DenseNet-CNN**         | 88.18        | 3.4614  / 8.3434  | threads=14 |
| Glove, DenseNet-DSA             | 87.66        | 6.9731  / -       |            |
| bpe BERT(4.8m), CNN             | 89.45        | 14.6978 / -       |            |
| bpe BERT(4.8m), CLS             | 89.59        | 14.0703 / -       |            |
| bpe BERT(4.8m), CNN             | 88.62        | 10.7023 / 73.4141 | del 8,9,10,11, threads=14 |
| bpe BERT(4.8m), CLS             | 88.92        | 9.3280  / 70.3232 | del 8,9,10,11, threads=14 |
| dha BERT(2.5m), CNN             | **89.96**    | 14.8779 / -       |            |
| dha BERT(2.5m), CLS             | 89.41        | 14.3664 / -       |            |
| dha BERT(2.5m), CNN             | 88.88        | 10.5157 / 72.7777 | del 8,9,10,11, threads=14                                        |
| dha BERT(2.5m), CLS             | 88.81        | 8.9836  / 68.4545 | del 8,9,10,11, threads=14, conda pytorch=1.2.0 50.7474ms         |
| dha BERT(2.5m), CLS             | 88.29        | 7.2027  / 53.6363 | del 6,7,8,9,10,11, threads=14, conda pytorch=1.2.0 38.3333ms     |
| dha BERT(2.5m), CLS             | 87.54        | 5.7645  / 36.8686 | del 4,5,6,7,8,9,10,11, threads=14, conda pytorch=1.2.0 28.2626ms |
| dha-bpe BERT(4m), CNN           | 89.07        | 14.9454 / -       |            |
| dha-bpe BERT(4m), CLS           | 89.01        | 12.7981 / -       |            |
| dha BERT(10m), CNN              | 89.08        | 15.3276 / -       |            |
| dha BERT(10m), CLS              | 89.25        | 12.7876 / -       |            |
| monologg ELECTRA-base , CNN     | 89.37        | 15.6362 / -       |            |
| monologg ELECTRA-base , CLS     | 89.30        | 14.4258 / -       |            |
| bpe ELECTRA-base(128.1m) , CNN  | -            | -       / -       |            |
| bpe ELECTRA-base(128.1m) , CLS  | 80.29        | 15.4886 / -       |            |

```
* GPU/CPU : Elapsed time/example(ms), GPU / CPU(pip 1.2.0)
* default batch size, learning rate : 128, 2e-4
```

- [HanBert-nsmc](https://github.com/monologg/HanBert-nsmc#results), [KoELECTRA](https://github.com/monologg/KoELECTRA)

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoELECTRA-Base    | 90.21        |
| XML-RoBERTa       | 89.49        |
| HanBert-54kN      | 90.16        |
| HanBert-54kN-IP   | 88.72        |
| KoBERT            | 89.63        |
| DistilKoBERT      | 88.41        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

- [KoBERT](https://github.com/SKTBrain/KoBERT#naver-sentiment-analysis)

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT            | 90.1         |


### GLOVE

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

* try again
INFO:__main__:[Accuracy] : 0.8744, 43715/49997
INFO:__main__:[Elapsed Time] : 522451ms, 10.447275782062565ms on average

* softmax masking
INFO:__main__:[Accuracy] : 0.8747, 43732/49997
INFO:__main__:[Elapsed Time] : 596904ms, 11.936694935594847ms on average

* iee_corpus_morph
$ python evaluate.py --config=configs/config-densenet-dsa-iee.json --data_dir=./data/iee_corpus_morph
```

</p>
</details>


### BERT(pytorch.all.bpe.4.8m_step)

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

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8931, 44653/49997
INFO:__main__:[Elapsed Time] : 672027ms, 13.439275142011361ms on average

INFO:__main__:[Accuracy] : 0.8959, 44790/49997
INFO:__main__:[Elapsed Time] : 703563ms, 14.07036562925034ms on average

  ** --bert_remove_layers=8,9,10,11
  INFO:__main__:[Accuracy] : 0.8892, 44457/49997
  INFO:__main__:[Elapsed Time] : 466825ms, 9.32800624049924ms on average
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

```

</p>
</details>


### BERT(pytorch.all.dha_s2.9.4_d2.9.27_bpe.4m_step)

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

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8901, 44503/49997
INFO:__main__:[Elapsed Time] : 639988ms, 12.798163853108248ms on average
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


### RoBERTa(kor-roberta-base.v1)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn
$ python preprocess.py --config=configs/config-roberta-cnn.json --bert_model_name_or_path=./embeddings/kor-roberta-base.v1 --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-roberta-cnn.json --bert_model_name_or_path=./embeddings/kor-roberta-base.v1 --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments

* enc_class=cls
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/kor-roberta-base.v1 --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-roberta-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

* enc_class=cls
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.5035, 25171/49997
INFO:__main__:[Elapsed Time] : 712393ms, 14.246719737579006ms on average

* something goes wrong!

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

* enc_class=cls
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8930, 44646/49997
INFO:__main__:[Elapsed Time] : 721693ms, 14.425894071525722ms on average

INFO:__main__:[Accuracy] : 0.8919, 44592/49997
INFO:__main__:[Elapsed Time] : 727322ms, 14.545743659492759ms on average

```

</p>
</details>

### ELECTRA(kor-electra-base-bpe-128-1m)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn
$ python preprocess.py --config=configs/config-electra-cnn.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-128-1m --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-electra-cnn.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-128-1m --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=20 --batch_size=64 --data_dir=./data/clova_sentiments

* enc_class=cls
$ python preprocess.py --config=configs/config-electra-cls.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-128-1m --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-electra-cls.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe-128-1m --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=20 --batch_size=64 --data_dir=./data/clova_sentiments 
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-electra-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint


* enc_class=cls
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8029, 40141/49997
INFO:__main__:[Elapsed Time] : 774480ms, 15.488699095927673ms on average

```

</p>
</details>



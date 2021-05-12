## Data

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
      - 원문의 'comment', 'hate' label만 사용
      - 'test.txt'는 제공하지 않으므로 'valid.txt'를 복사해서 사용.
      - data augmentation(distillation) 등에 활용하기 위해서 'unlabeled' 데이터도 복사.
        - 데이터 사이즈가 제법 크기 때문에, git에 추가하지 않고, 다운받아서 사용.

    - './data/korean_hate_speech_morph'
      - `형태소분석기 tokenizer`를 적용한 데이터.

    - './data/korean_bias_speech/'
      - 원문의 'comment', 'bias' label만 사용
    - './data/korean_bias_speech_morph'
      - `형태소분석기 tokenizer`를 적용한 데이터.

## Pretrained models

#### GloVe

- GloVe
  - [Standford GloVe code](https://github.com/stanfordnlp/GloVe)를 이용해서 학습.
  - 한국어 문서 데이터 준비.
    - 다양한 문서 데이터(위키, 백과, 뉴스, 블로그 등등)를 크롤링.
  - 형태소분석기 tokenizer를 적용해서 형태소 단위로 변경한 데이터를 이용해서 학습 진행.
  - `kor.glove.300k.300d.txt` (inhouse)

#### BERT

- bpe / dha / dha_bpe BERT, BERT-large
  - [google original tf code](https://github.com/google-research/bert)를 이용해서 학습.
  - 한국어 문서 데이터 준비.
    - 위 한국어 GloVe 학습에 사용한 데이터를 그대로 이용.
  - `character-level bpe`
    - vocab.txt는 [sentencepiece](https://github.com/google/sentencepiece)를 이용해서 생성.
    - `kor-bert-base-bpe.v1`, `kor-bert-large-bpe.v1, v3` (inhouse)
  - `character-level bpe + 형태소분석기`
    - ex) `kor-bert-base-dha_bpe.v1, v3`, `kor-bert-large-dha_bpe.v1, v3` (inhouse)
  - `형태소분석기`
    - `kor-bert-base-dha.v1, v2` (inhouse)

- KcBERT 
  - [KcBERT](https://github.com/Beomi/KcBERT)
    - `kcbert-base`, `kcbert-large`

#### DistilBERT

- DistilBERT
  - [training-distilbert](https://github.com/dsindex/transformers_examples#training-distilbert)
  - 한국어 문서 데이터 준비.
    - 위 한국어 GloVe 학습에 사용한 데이터를 그대로 이용.
  - `kor-distil-bpe-bert.v1`, `kor-distil-dha-bert.v1` (inhouse)
  - `kor-distil-wp-bert.v1` (inhouse)
    - `koelectra-base-v3-discriminator`를 distillation. 학습데이터는 동일.

- mDistilBERT
  - from [huggingface.co/models](https://huggingface.co/models)
  - `distilbert-base-multilingual-cased` 

#### ELECTRA

- KoELECTRA-Base
  - [KoELECTRA](https://github.com/monologg/KoELECTRA)
    - `monologg/koelectra-base-v1-discriminator`, `monologg/koelectra-base-v3-discriminator`

- LM-KOR-ELECTRA
  - [LM-kor](https://github.com/kiyoungkim1/LM-kor)
    - `kykim/electra-kor-base`

- ELECTRA-base
  - [electra](https://github.com/dsindex/electra#pretraining-electra)를 이용해서 학습.
  - 한국어 문서 데이터 준비.
    - 위 한국어 GloVe 학습에 사용한 데이터를 그대로 이용.
  - [README.md](https://github.com/dsindex/electra/blob/master/README.md)
  - [train.sh](https://github.com/dsindex/electra/blob/master/train.sh)
  - `kor-electra-base-bpe.v1`, `kor-electra-base-dhaToken1.large` (inhouse)

#### RoBERTa

- RoBERTa-base
  - [huggingface](https://huggingface.co/blog/how-to-train)를 이용한 학습
    - 한국어 문서 데이터 준비.
      - 위 한국어 GloVe 학습에 사용한 데이터를 그대로 이용.
    - `kor-roberta-base-bbpe` (inhouse)

- XLM-RoBERTa-base, XML-RoBERTa-large
  - from [huggingface.co/models](https://huggingface.co/models)
  - `xlm-roberta-base`, `xlm-roberta-large`

#### Funnel

- Funnel-base
  - from [LMKor](https://github.com/kiyoungkim1/LMkor)
  - `funnel-kor-base`

#### BART

- KoBART-base
  - from [KoBART](https://github.com/SKT-AI/KoBART)

#### GPT

- KoGPT2
  - from [KoGPT2](https://github.com/SKT-AI/KoGPT2)



## NMSC data

- iclassifier

|                                           | Accuracy (%) | GPU / CPU         | Etc        |
| ----------------------------------------- | ------------ | ----------------- | ---------- |
| GloVe, GNB                                | 74.68        | 1.3568  / -       |            |
| GloVe, CNN                                | 87.31        | 1.9479  / 3.5353  | threads=14 |
| **GloVe, DenseNet-CNN**                   | 88.18        | 3.4614  / 8.3434  | threads=14 |
| GloVe, DenseNet-DSA                       | 87.66        | 6.9731  / -       |            |
| DistilFromBERT, GloVe, DenseNet-CNN       | 89.21        | 3.5383  / -       | augmented, from 'dha BERT(v1), CLS'           |
| DistilFromBERT, GloVe, DenseNet-CNN       | 89.14        | 3.6146  / -       | augmented, from 'dha-bpe BERT-large(v1), CNN' |
| DistilFromBERT, dha DistilBERT(v1), CLS   | 90.19        | 8.9599  / -       | augmented, from 'dha-bpe BERT-large(v1), CNN' |
| bpe DistilBERT(v1), CNN                   | 88.39        | 9.6396  / -       | threads=14 |
| bpe DistilBERT(v1), CLS                   | 88.55        | 8.2834  / -       | threads=14 |
| wp  DistilBERT(v1), CNN                   | 88.04        | 8.7733  / -       |            |
| wp  DistilBERT(v1), CLS                   | 88.08        | 8.0111  / -       |            |
| mDistilBERT, CLS                          | 86.55        | 8.0584  / -       |            |
| bpe BERT(v1), CNN                         | 90.11        | 16.5453 / -       |            |
| bpe BERT(v1), CLS                         | 89.91        | 14.9586 / -       |            |
| bpe BERT(v1), CNN                         | 88.62        | 10.7023 / 73.4141 | del 8,9,10,11, threads=14 |
| bpe BERT(v1), CLS                         | 88.92        | 9.3280  / 70.3232 | del 8,9,10,11, threads=14 |
| bpe BERT-large(v1), CNN                   | 89.85        | 24.4099 / -       |            |
| bpe BERT-large(v1), CLS                   | 89.78        | 22.6002 / -       |            |
| bpe BERT-large(v3), CNN                   | 89.45        | 26.5318 / -       |            |
| bpe BERT-large(v3), CLS                   | 89.06        | 25.0526 / -       |            |
| KcBERT-base , CNN                         | 90.10        | 14.2056 / -       |            |
| KcBERT-base , CLS                         | 90.23        | 13.5712 / -       |            |
| KcBERT-large , CNN                        | 91.26        | 24.2121 / -       |            |
| KcBERT-large , CLS                        | 91.36        | 22.4859 / -       |            |
| dha DistilBERT(v1), CNN                   | 88.72        | 11.4488 / -       |            |
| dha DistilBERT(v1), CLS                   | 88.51        | 7.5299  / -       |            |
| dha BERT(v1), CNN                         | 90.25        | 15.5738 / -       |            |
| dha BERT(v1), CLS                         | 90.18        | 13.3390 / -       |            |
| dha BERT(v1), CNN                         | 88.88        | 10.5157 / 72.7777 | del 8,9,10,11, threads=14         |
| dha BERT(v1), CLS                         | 88.81        | 8.9836  / 68.4545 | del 8,9,10,11, threads=14         |
| dha BERT(v1), CLS                         | 88.29        | 7.2027  / 53.6363 | del 6,7,8,9,10,11, threads=14     |
| dha BERT(v1), CLS                         | 87.54        | 5.7645  / 36.8686 | del 4,5,6,7,8,9,10,11, threads=14 |
| dha BERT(v2), CNN                         | 89.08        | 15.3276 / -       |            |
| dha BERT(v2), CLS                         | 89.25        | 12.7876 / -       |            |
| dha-bpe BERT(v1), CNN                     | 89.07        | 14.9454 / -       |            |
| dha-bpe BERT(v1), CLS                     | 89.01        | 12.7981 / -       |            |
| dha-bpe BERT(v3), CNN                     | 89.91        | 14.8520 / -       |            |
| dha-bpe BERT(v3), CLS                     | 89.93        | 13.6896 / -       |            |
| dha-bpe BERT-large(v1), CNN               | 90.84        | 24.5095 / -       |            |
| dha-bpe BERT-large(v1), CLS               | 90.68        | 22.9305 / -       |            |
| dha-bpe BERT-large(v3), CNN               | 90.44        | 28.7014 / -       |            |
| dha-bpe BERT-large(v3), CLS               | 90.57        | 25.5458 / -       |            |
| KoELECTRA-Base-v1, CNN                    | 89.51        | 15.5452 / -       |            |
| KoELECTRA-Base-v1, CLS                    | 89.63        | 14.2667 / -       |            |
| KoELECTRA-Base-v3, CNN                    | 90.72        | 15.3168 / -       |            |
| KoELECTRA-Base-v3, CLS                    | 90.66        | 13.7968 / -       |            |
| LM-KOR-ELECTRA, CNN                       | 90.52        | 15.8000 / -       |            |
| LM-KOR-ELECTRA, CLS                       | 91.04        | 14.2696 / -       |            |
| bpe ELECTRA-base(v1) , CNN                | 89.59        | 15.8888 / -       |            |
| bpe ELECTRA-base(v1) , CLS                | 89.59        | 14.3914 / -       |            |
| dhaToken1.large ELECTRA-base , CLS        | 90.88        | 14.3333 / -       |            |
| RoBERTa-base , CNN                        | 90.42        | 14.9544 / -       |            |
| RoBERTa-base , CLS                        | 90.34        | 13.8556 / -       |            |
| XLM-RoBERTa-base , CLS                    | 89.98        | 14.8101 / -       |            |
| XLM-RoBERTa-large , CLS                   | 91.05        | 25.1067 / -       |            |
| Funnel-base , CLS                         | **91.51**    | 41.8325 / -       |            |
| KoBART-base , CLS                         | 89.57        | 18.9681 / -       |            |
| KoGPT2-v2 , CLS                           | 88.57        | 16.0536 / -       |            |

```
* GPU/CPU : Elapsed time/example(ms), GPU / CPU
* default batch size, learning rate, n_ctx(max_seq_length) : 128, 2e-4, 100
```

- [HanBert-nsmc](https://github.com/monologg/HanBert-nsmc#results), [KoELECTRA](https://github.com/monologg/KoELECTRA), [LM-kor](https://github.com/kiyoungkim1/LM-kor)

|                   | Accuracy (%) | Etc        |
| ----------------- | ------------ | ---------- |
| KoELECTRA-Base-v1 | 90.33        |            |
| KoELECTRA-Base-v2 | 89.56        |            |
| KoELECTRA-Base-v3 | 90.63        |            |
| electra-kor-base  | 91.29        |            |
| funnel-kor-base   | **91.36**    |            |
| XML-RoBERTa       | 89.03        |            |
| HanBERT           | 90.06        |            |
| DistilKoBERT      | 88.60        |            |
| Bert-Multilingual | 87.07        |            |
| FastText          | 85.50        |            |

- [KoBERT](https://github.com/SKTBrain/KoBERT#naver-sentiment-analysis), [KoGPT2](https://github.com/SKT-AI/KoGPT2), [KoBART](https://github.com/SKT-AI/KoBART)

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT            | 90.1         |
| KoGPT2            | **93.3**     |
| KoBART            | 90.24        |

- [aisolab/nlp_classification](https://github.com/aisolab/nlp_classification)
  - 비교를 위해서, 여기에서는 데이터를 동일하게 맞추고 재실험.
  - `--epoch=10 --learning_rate=5e-4 --batch_size=128`

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT(STKBERT)   | 89.35        |
| ETRIBERT          | **89.99**    |


#### GloVe

<details><summary><b>enc_class=gnb</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-glove-gnb.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-glove-gnb.json --data_dir=data/clova_sentiments_morph --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-gnb.json --data_dir=./data/clova_sentiments_morph 
INFO:__main__:[Accuracy] : 0.7468, 37339/49997
INFO:__main__:[Elapsed Time] : 67916.67175292969ms, 1.3568544208512268ms on average
```

</p>
</details>


<details><summary><b>enc_class=cnn</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-glove-cnn.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-glove-cnn.json --data_dir=data/clova_sentiments_morph --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-cnn.json --data_dir=./data/clova_sentiments_morph 

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
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph 

* iee_corpus_morph
$ python preprocess.py --config=configs/config-densenet-cnn-iee.json --data_dir=data/iee_corpus_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn-iee.json --data_dir=data/iee_corpus_morph 
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/clova_sentiments_morph 

INFO:__main__:[Accuracy] : 0.8818, 44087/49997
INFO:__main__:[Elapsed Time] : 173152ms, 3.4614969197535803ms on average

INFO:__main__:[Accuracy] : 0.8799, 43991/49997
INFO:__main__:[Elapsed Time] : 179413.97333145142ms, 3.58712682724762ms on average

*  --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8804, 44017/49997
INFO:__main__:[Elapsed Time] : 160672.52135276794ms, 3.211939556139528ms on average

INFO:__main__:[Accuracy] : 0.8803, 44012/49997
INFO:__main__:[Elapsed Time] : 161198.18258285522ms, 3.2225625831057085ms on average

* hyper-parameter search by optuna
*  --warmup_epoch=0 --weight_decay=0.0 --lr=0.00014915118702339241 --batch_size=128 --seed=34 --epoch=32
INFO:__main__:[Accuracy] : 0.8822, 44108/49997
INFO:__main__:[Elapsed Time] : 189978.82962226868ms, 3.7981278658466384ms on average

INFO:__main__:[study.best_params] : {'lr': 5.124781058611912e-05, 'batch_size': 64, 'seed': 25, 'epochs': 57}
INFO:__main__:[study.best_value] : 0.8811528691721503

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
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph 

* iee_corpus_morph
$ python preprocess.py --config=configs/config-densenet-dsa-iee.json --data_dir=data/iee_corpus_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-dsa-iee.json --data_dir=data/iee_corpus_morph --batch_size=256
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

*  --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8735, 43670/49997
INFO:__main__:[Elapsed Time] : 410616.1913871765ms, 8.211271306382855ms on average

* iee_corpus_morph
$ python evaluate.py --config=configs/config-densenet-dsa-iee.json --data_dir=./data/iee_corpus_morph
```

</p>
</details>


#### BERT(kor-bert-base-bpe, kor-bert-large-bpe, kor-distil-bpe-bert, distilbert-base-multilingual-cased, kcbert-base, kcbert-large)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments/

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments/
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

**  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9011, 45053/49997
INFO:__main__:[Elapsed Time] : 827306ms, 16.545303624289943ms on average

** --configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8839, 44190/49997
INFO:__main__:[Elapsed Time] : 482054.96978759766ms, 9.639614557722052ms on average

** --configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-wp-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8804, 44018/49997
INFO:__main__:[Elapsed Time] : 438734.2314720154ms, 8.773349530125191ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v1  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30
INFO:__main__:[Accuracy] : 0.8985, 44923/49997
INFO:__main__:[Elapsed Time] : 1220545.0494289398ms, 24.40995301749384ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v3  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=20 --patience=4
INFO:__main__:[Accuracy] : 0.8945, 44722/49997
INFO:__main__:[Elapsed Time] : 1326633.1841945648ms, 26.53188067003673ms on average

** --bert_model_name_or_path=./embeddings/kcbert-base  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9010, 45047/49997
INFO:__main__:[Elapsed Time] : 710366.4381504059ms, 14.20562694488368ms on average

** --bert_model_name_or_path=./embeddings/kcbert-large  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9102, 45509/49997
INFO:__main__:[Elapsed Time] : 1230672.8971004486ms, 24.612369814131945ms on average

** --bert_model_name_or_path=./embeddings/kcbert-large  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=1e-5
INFO:__main__:[Accuracy] : 0.9126, 45627/49997
INFO:__main__:[Elapsed Time] : 1210648.3128070831ms, 24.212181653983308ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8931, 44653/49997
INFO:__main__:[Elapsed Time] : 672027ms, 13.439275142011361ms on average

INFO:__main__:[Accuracy] : 0.8959, 44790/49997
INFO:__main__:[Elapsed Time] : 703563ms, 14.07036562925034ms on average

**  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8991, 44952/49997
INFO:__main__:[Elapsed Time] : 747975ms, 14.958656692535403ms on average

** n_ctx=64,   --warmup_epoch=1 --weight_decay=0.0 --seed=0 --epoch=30
INFO:__main__:[Accuracy] : 0.8970, 44848/49997
INFO:__main__:[Elapsed Time] : 595030.0581455231ms, 11.899778242826137ms on average

** --bert_remove_layers=8,9,10,11
INFO:__main__:[Accuracy] : 0.8892, 44457/49997
INFO:__main__:[Elapsed Time] : 466825ms, 9.32800624049924ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8855, 44271/49997
INFO:__main__:[Elapsed Time] : 414233.4134578705ms, 8.283499222067283ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-wp-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8808, 44036/49997
INFO:__main__:[Elapsed Time] : 400613.59667778015ms, 8.011125518718865ms on average

** --configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/distilbert-base-multilingual-cased --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.8655, 43273/49997
INFO:__main__:[Elapsed Time] : 402990.97418785095ms, 8.058487663212581ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v1  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30
INFO:__main__:[Accuracy] : 0.8978, 44885/49997
INFO:__main__:[Elapsed Time] : 1130058.8986873627ms, 22.60026191461467ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-bpe.v3  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=20 --patience=4
INFO:__main__:[Accuracy] : 0.8906, 44526/49997
INFO:__main__:[Elapsed Time] : 1252674.3474006653ms, 25.05268840963759ms on average

** --bert_model_name_or_path=./embeddings/kcbert-base  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9023, 45110/49997
INFO:__main__:[Elapsed Time] : 678645.0374126434ms, 13.571247471572018ms on average

** --bert_model_name_or_path=./embeddings/kcbert-large  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --lr=1e-5
INFO:__main__:[Accuracy] : 0.9136, 45677/49997
INFO:__main__:[Elapsed Time] : 1124363.7096881866ms, 22.48598377803238ms on average

```

</p>
</details>


#### BERT(kor-bert-base-dha, kor-distil-dha-bert)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph/

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=3 --batch_size=64 --data_dir=./data/clova_sentiments_morph/
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

**  --warmup_epoch=0 --weight_decay=0.0 --epoch=20
INFO:__main__:[Accuracy] : 0.9025, 45123/49997
INFO:__main__:[Elapsed Time] : 778762ms, 15.573805904472358ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v2
INFO:__main__:[Accuracy] : 0.8908, 44536/49997
INFO:__main__:[Elapsed Time] : 766457ms, 15.327646211696935ms on average

** --config=configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=20
(1) epoch_0
INFO:__main__:[Accuracy] : 0.8872, 44358/49997
INFO:__main__:[Elapsed Time] : 572516.1464214325ms, 11.448896296530382ms on average

(2) epoch_2
INFO:__main__:[Accuracy] : 0.8864, 44315/49997
INFO:__main__:[Elapsed Time] : 430667.7966117859ms, 8.612132680941624ms on average

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

**  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
INFO:__main__:[Accuracy] : 0.9018, 45089/49997
INFO:__main__:[Elapsed Time] : 666997.1199035645ms, 13.339050636929372ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v2
INFO:__main__:[Accuracy] : 0.8925, 44622/49997
INFO:__main__:[Elapsed Time] : 639463ms, 12.787603008240659ms on average

** --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1  --warmup_epoch=0 --weight_decay=0.0 --epoch=30
(1) epoch_0
INFO:__main__:[Accuracy] : 0.8847, 44234/49997
INFO:__main__:[Elapsed Time] : 486852.8757095337ms, 9.7360708339852ms on average

(2) epoch_2
INFO:__main__:[Accuracy] : 0.8851, 44252/49997
INFO:__main__:[Elapsed Time] : 376557.34610557556ms, 7.529911446336728ms on average

```

</p>
</details>


#### BERT(kor-bert-base-dha_bpe, kor-bert-large-dha_bpe)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1 --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1 --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph

* enc_class=cls

$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v1 --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/clova_sentiments_morph
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8907, 44533/49997
INFO:__main__:[Elapsed Time] : 747351ms, 14.945475638051045ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v3 --epoch=30 --warmup_epoch=0 --weight_decay=0.0 --patience 4
INFO:__main__:[Accuracy] : 0.8991, 44950/49997
INFO:__main__:[Elapsed Time] : 742653.1262397766ms, 14.852031124754996ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30
INFO:__main__:[Accuracy] : 0.9084, 45417/49997
INFO:__main__:[Elapsed Time] : 1225501.6918182373ms, 24.509510690474073ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v3  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=20 --patience 4
INFO:__main__:[Accuracy] : 0.9044, 45219/49997
INFO:__main__:[Elapsed Time] : 1435084.29813385ms, 28.701459281835014ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8901, 44503/49997
INFO:__main__:[Elapsed Time] : 639988ms, 12.798163853108248ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-base-dha_bpe.v3 --epoch=30 --warmup_epoch=0 --weight_decay=0.0 --patience 4
INFO:__main__:[Accuracy] : 0.8993, 44964/49997
INFO:__main__:[Elapsed Time] : 684546.1826324463ms, 13.68961429412827ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=30
INFO:__main__:[Accuracy] : 0.9068, 45337/49997
INFO:__main__:[Elapsed Time] : 1146557.6510429382ms, 22.930594570818535ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v3  --warmup_epoch=0 --weight_decay=0.0 --lr=1e-5 --epoch=20 --patience 4
INFO:__main__:[Accuracy] : 0.9057, 45281/49997
INFO:__main__:[Elapsed Time] : 1277329.5888900757ms, 25.545897660460298ms on average

```

</p>
</details>


#### ELECTRA(koelectra-base-discriminator, electra-kor-base)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json

* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64 --data_dir=./data/clova_sentiments

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/koelectra-base-v1-discriminator --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --batch_size=64 --data_dir=./data/clova_sentiments
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.8937, 44684/49997
INFO:__main__:[Elapsed Time] : 784375ms, 15.636230898471878ms on average

**  --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8951, 44750/49997
INFO:__main__:[Elapsed Time] : 777338ms, 15.54522361788943ms on average

** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator  --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9072, 45356/49997
INFO:__main__:[Elapsed Time] : 765895.2033519745ms, 15.316871435453972ms on average

** --bert_model_name_or_path=./embeddings/electra-kor-base --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9052, 45257/49997
INFO:__main__:[Elapsed Time] : 790089.4966125488ms, 15.800023858132711ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.8930, 44646/49997
INFO:__main__:[Elapsed Time] : 721693ms, 14.425894071525722ms on average

**  --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.8963, 44814/49997
INFO:__main__:[Elapsed Time] : 713403ms, 14.266721337707017ms on average

** --bert_model_name_or_path=./embeddings/koelectra-base-v3-discriminator  --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9066, 45325/49997
INFO:__main__:[Elapsed Time] : 689906.133890152ms, 13.796895782950783ms on average

** --bert_model_name_or_path=./embeddings/electra-kor-base --lr=5e-5 --epoch=20 --batch_size=64 --warmup_epoch=0 --weight_decay=0.0
INFO:__main__:[Accuracy] : 0.9104, 45516/49997
INFO:__main__:[Elapsed Time] : 713545.1235771179ms, 14.269641169796696ms on average

```

</p>
</details>

#### ELECTRA(kor-electra-base-bpe, kor-electra-base-dhaToken1.large)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json

* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1 --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1 --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments  --warmup_epoch=0 --weight_decay=0.0 

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1 --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-electra-base-bpe.v1 --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments  --warmup_epoch=0 --weight_decay=0.0
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.8959, 44792/49997
INFO:__main__:[Elapsed Time] : 794492.9943084717ms, 15.88887755456319ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.8959, 44790/49997
INFO:__main__:[Elapsed Time] : 719611.6433143616ms, 14.391430078177311ms on average

** --bert_model_name_or_path=./embeddings/kor-electra-base-dhaToken1.large
INFO:__main__:[Accuracy] : 0.9088, 45436/49997
INFO:__main__:[Elapsed Time] : 716717.779636383ms, 14.333377548080128ms on average

```

</p>
</details>

#### RoBERTa(kor-roberta-base-bbpe, xlm-roberta-base, xlm-roberta-large)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-roberta-cnn.json --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-roberta-cnn.json --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments  --warmup_epoch=0 --weight_decay=0.0 

* enc_class=cls

$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/kor-roberta-base-bbpe --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments  --warmup_epoch=0 --weight_decay=0.0
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-roberta-cnn.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9042, 45207/49997
INFO:__main__:[Elapsed Time] : 747780.4193496704ms, 14.954407292944458ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9034, 45168/49997
INFO:__main__:[Elapsed Time] : 692853.7294864655ms, 13.855628213249918ms on average

** --bert_model_name_or_path=./embeddings/xlm-roberta-base
INFO:__main__:[Accuracy] : 0.8998, 44986/49997
INFO:__main__:[Elapsed Time] : 740546.2129116058ms, 14.810195497745838ms on average

** --bert_model_name_or_path=./embeddings/xlm-roberta-large
INFO:__main__:[Accuracy] : 0.9105, 45523/49997
INFO:__main__:[Elapsed Time] : 1255374.0434646606ms, 25.106745840540047ms on average
```

</p>
</details>

#### Funnel(funnel-kor-base)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* share config-bert-*.json

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/funnel-kor-base --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/funnel-kor-base --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments  --warmup_epoch=0 --weight_decay=0.0
```

- evaluation
```
* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=./data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9151, 45751/49997
INFO:__main__:[Elapsed Time] : 2091641.2904262543ms, 41.832550390881245ms on average

```

</p>
</details>

#### BART(kobart)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
$ vi configs/config-bart-cls.json
    "use_kobart": true,

* enc_class=cls

$ python preprocess.py --config=configs/config-bart-cls.json --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bart-cls.json --save_path=pytorch-model.pt --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments
```

- evaluation
```
* enc_class=cls

$ python evaluate.py --config=configs/config-bart-cls.json --data_dir=./data/clova_sentiments --model_path=pytorch-model.pt
INFO:__main__:[Accuracy] : 0.8957, 44781/49997
INFO:__main__:[Elapsed Time] : 948470.7288742065ms, 18.96814218339219ms on average

```

</p>
</details>

#### GPT(kogpt2-base-v2)
 
<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
# n_ctx(max_seq_length) : 64

$ python preprocess.py --config=configs/config-gpt-cls.json --bert_model_name_or_path='skt/kogpt2-base-v2' --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-gpt-cls.json --bert_model_name_or_path='skt/kogpt2-base-v2' --save_path=pytorch-model.pt --lr=5e-5 --epoch=10 --batch_size=128 --warmup_ratio=0.1 --seed=7874 --data_dir=./data/clova_sentiments

```

- evaluation
```
* enc_class=cls

$ python evaluate.py --config=configs/config-gpt-cls.json --data_dir=./data/clova_sentiments --bert_output_dir='skt/kogpt2-base-v2' --model_path=pytorch-model.pt

INFO:__main__:[Accuracy] : 0.8857, 44281/49997
INFO:__main__:[Elapsed Time] : 802752.3436546326ms, 16.053675255667486ms on average
```

</p>
</details>


## korean-hate-speech data

- iclassifier

|                                           | Bias Accuracy (%) | Hate Accuracy (%) | GPU / CPU         | Etc                                                |
| ----------------------------------------- | ----------------- | ----------------- | ----------------- | -------------------------------------------------- |
| GloVe, GNB                                | 72.61             | 35.24             | 1.4223  / -       | failed to train for Bias data                      |
| GloVe, CNN                                | 79.62             | 60.72             | 2.1536  / -       | LabelSmoothingCrossEntropy for Bias                |
| GloVe, DenseNet-CNN                       | 84.08             | 61.78             | 3.9363  / -       | LabelSmoothingCrossEntropy for Bias                |
| GloVe, DenseNet-DSA                       | 84.29             | 59.87             | 9.1583  / -       | LabelSmoothingCrossEntropy for Bias                |
| Augmentation, GloVe, GNB                  | 73.25             | 38.85             | 1.3125  / -       |                                                    |
| Augmentation, GloVe, CNN                  | 80.68             | 59.45             | 1.7384  / -       |                                                    |
| Augmentation, GloVe, DenseNet-CNN         | 81.95             | 59.87             | 3.6013  / -       |                                                    |
| Augmentation, GloVe, DenseNet-DSA         | 84.72             | 59.02             | 7.4648  / -       |                                                    |
| **DistilFromBERT, GloVe, DenseNet-CNN**   | 83.65             | 64.97             | 3.8358  / -       | from 'dha BERT(v1), CNN', augmented                |
| DistilFromBERT, GloVe, DenseNet-CNN       | **85.56**         | 66.67             | 3.6249  / -       | from 'dha BERT(v1), CNN', augmented, unlabeled data used        |
| DistilFromBERT, GloVe, DenseNet-CNN       | 84.08             | 62.63             | 3.8700  / -       | from 'bpe BERT(v1), CNN', no augmentation          |
| DistilFromBERT, dha DistilBERT(v1), CNN   | 85.56             | 63.91             | 9.6239  / -       | from 'dha BERT(v1), CNN', unlabeled data used      |
| DistilFromBERT, bpe DistilBERT(v1), CNN   | 84.29             | 63.69             | 8.6725  / -       | from 'bpe BERT(v1), CNN', no augmentation          |
| DistilFromBERT, bpe DistilBERT(v1), CNN   | 83.44             | 64.12             | 8.5794  / -       | from 'bpe BERT(v1), CNN', no augmentation, unlabeled data used  |
| dha DistilBERT(v1), CNN                   | 82.59             | 64.54             | 14.7450 / -       |                                                    |
| dha DistilBERT(v1), CLS                   | 83.23             | 62.42             | 13.0598 / -       |                                                    |
| dha BERT(v1), CNN                         | 84.08             | **67.09**         | 15.8797 / -       |                                                    |
| dha BERT(v1), CLS                         | 82.80             | 64.76             | 12.8167 / -       |                                                    |
| dha BERT(v1)-NSMC, CNN                    | 83.44             | 65.61             | 14.8844 / -       | finetuned with 'dha BERT(v1), CLS', NSMC           |
| dha BERT(v1)-NSMC, CLS                    | 82.80             | 66.03             | 13.3459 / -       | finetuned with 'dha BERT(v1), CLS', NSMC           |
| dha-bpe BERT-large(v1), CNN               | 83.86             | 66.03             | 33.4405 / -       |                                                    |
| dha-bpe BERT-large(v1), CLS               | 83.86             | 66.67             | 28.3876 / -       |                                                    |
| bpe DistilBERT(v1), CNN                   | 82.38             | 60.93             | 8.7683  / -       |                                                    |
| bpe DistilBERT(v1), CLS                   | 81.53             | 61.36             | 7.6983  / -       |                                                    |
| bpe BERT(v1), CNN                         | 82.80             | 63.27             | 15.0740 / -       |                                                    |
| bpe BERT(v1), CLS                         | 82.38             | 63.69             | 13.1576 / -       |                                                    |

```
* GPU/CPU : Elapsed time/example(ms), GPU / CPU
* default batch size, learning rate, n_ctx(max_seq_length) : 128, 2e-4, 100
* korean_bias_speech 데이터의 경우는 'none' class의 비율이 높아서 bias가 있는 편이다. 
  (korean_hate_speech 데이터에 비해 accuarcy가 많이 높은 원인도 여기에 있을듯)
  따라서, average F1을 지표로 사용하는 것이 좀 더 좋겠지만, 편의상 accuracy를 사용했음.
```

- [korean-hate-speech-koelectra](https://github.com/monologg/korean-hate-speech-koelectra)

| (Weighted F1)     | Bias F1 (%)       | Hate F1 (%)       | Etc                                  |
| ----------------- | ----------------- | ----------------- | ------------------------------------ |
| KoELECTRA-base    | **82.28**         | **67.25**         | with title, bias/hate joint training |


#### GloVe

<details><summary><b>enc_class=glove-gnb</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-glove-gnb.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-glove-gnb.json --data_dir=data/korean_hate_speech_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-gnb.pt

```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-gnb.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-gnb.pt
INFO:__main__:[Accuracy] : 0.3397,   160/  471
INFO:__main__:[Elapsed Time] : 759.1080665588379ms, 1.4223291518840384ms on average

** --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.3524,   166/  471
INFO:__main__:[Elapsed Time] : 780.1830768585205ms, 1.4809202640614612ms on average

** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.7261,   342/  471
INFO:__main__:[Elapsed Time] : 673.8839149475098ms, 1.2677735470710916ms on average

=> 학습 결과가 전부 'none' class를 찍는 문제. 학습이 안됨

** --data_dir=./data/korean_bias_speech_morph --criterion=LabelSmoothingCrossEntropy

=> 여전히 학습이 안됨

** augmentation
$ python augment_data.py --input data/korean_hate_speech/train.txt --output data/korean_hate_speech_morph/augmented.txt --analyzer=npc --n_iter=5 --max_ng=3 --preserve_label --parallel
$ python preprocess.py --config=configs/config-glove-gnb.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt --augmented --augmented_filename augmented.txt
$ python train.py --config=configs/config-glove-gnb.json --data_dir=data/korean_hate_speech_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-gnb.pt --augmented --criterion CrossEntropyLoss
$ python evaluate.py --config=configs/config-glove-gnb.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-gnb.pt
INFO:__main__:[Accuracy] : 0.3885,   183/  471
INFO:__main__:[Elapsed Time] : 703.909158706665ms, 1.3125252216420276ms on average

*** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.7325,   345/  471
INFO:__main__:[Elapsed Time] : 652.2922515869141ms, 1.2430145385417533ms on average

```

</p>
</details>

<details><summary><b>enc_class=glove-cnn</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-glove-cnn.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-glove-cnn.json --data_dir=data/korean_hate_speech_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-cnn.pt

```

- evaluation
```
$ python evaluate.py --config=configs/config-glove-cnn.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-cnn.pt
INFO:__main__:[Accuracy] : 0.6072,   286/  471
INFO:__main__:[Elapsed Time] : 1125.8411407470703ms, 2.2128495764225087ms on average

** --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.5860,   276/  471
INFO:__main__:[Elapsed Time] : 1131.6962242126465ms, 2.2415470569691758ms on average

** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.7261,   342/  471
INFO:__main__:[Elapsed Time] : 1100.193738937378ms, 2.101261057752244ms on average

=> 학습 결과가 전부 'none' class를 찍는 문제. 학습이 안됨

** --data_dir=./data/korean_bias_speech_morph --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.7962,   375/  471
INFO:__main__:[Elapsed Time] : 1090.4510021209717ms, 2.1536132122607943ms on average

** augmentation
INFO:__main__:[Accuracy] : 0.5945,   280/  471
INFO:__main__:[Elapsed Time] : 904.7646522521973ms, 1.7384361713490588ms on average

*** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8068,   380/  471
INFO:__main__:[Elapsed Time] : 1002.643346786499ms, 1.9655851607627057ms on average

```

</p>
</details>


<details><summary><b>enc_class=densenet-cnn</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/korean_hate_speech_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-cnn.pt

```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-cnn.pt
INFO:__main__:[Accuracy] : 0.6178,   291/  471
INFO:__main__:[Elapsed Time] : 1848.5705852508545ms, 3.760207967555269ms on average

** --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.6178,   291/  471
INFO:__main__:[Elapsed Time] : 2014.061689376831ms, 4.075388198203229ms on average

** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.7261,   342/  471
INFO:__main__:[Elapsed Time] : 2106.7402362823486ms, 4.309217473293873ms on average

=> 학습 결과가 전부 'none' class를 찍는 문제. 학습이 안됨

** --data_dir=./data/korean_bias_speech_morph --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.8408,   396/  471
INFO:__main__:[Elapsed Time] : 1953.4156322479248ms, 3.9363637883612452ms on average

** augmentation
INFO:__main__:[Accuracy] : 0.5987,   282/  471
INFO:__main__:[Elapsed Time] : 1770.780086517334ms, 3.6013182173383997ms on average

*** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8195,   386/  471
INFO:__main__:[Elapsed Time] : 1687.3586177825928ms, 3.401884119561378ms on average

```

</p>
</details>


<details><summary><b>enc_class=densenet-dsa</b></summary>
<p>

- train
```
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/korean_hate_speech_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-dsa.pt

```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-dsa.pt
INFO:__main__:[Accuracy] : 0.5987,   282/  471
INFO:__main__:[Elapsed Time] : 6090.857982635498ms, 8.307155142439173ms on average

** --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.5860,   276/  471
INFO:__main__:[Elapsed Time] : 4470.3404903411865ms, 9.266544402913844ms on average

** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.7261,   342/  471
INFO:__main__:[Elapsed Time] : 6214.275121688843ms, 8.474696950709566ms on average

=> 학습 결과가 전부 'none' class를 찍는 문제. 학습이 안됨

** --data_dir=./data/korean_bias_speech_morph --criterion=LabelSmoothingCrossEntropy
INFO:__main__:[Accuracy] : 0.8429,   397/  471
INFO:__main__:[Elapsed Time] : 4414.142608642578ms, 9.158381502679054ms on average

** augmentation
INFO:__main__:[Accuracy] : 0.5902,   278/  471
INFO:__main__:[Elapsed Time] : 3587.0931148529053ms, 7.464842086142682ms on average

*** --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8471,   399/  471
INFO:__main__:[Elapsed Time] : 3581.5823078155518ms, 7.451774718913626ms on average

```

</p>
</details>



#### BERT(kor-bert-base-dha, kor-bert-large-dha_bpe)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --data_dir=./data/korean_hate_speech_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64  --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech_morph --save_path=pytorch-model-kor-bert.pt

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --data_dir=./data/korean_hate_speech_morph
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-dha.v1 --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64  --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech_morph --save_path=pytorch-model-kor-bert.pt
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/korean_hate_speech_morph --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
INFO:__main__:[Accuracy] : 0.6709,   316/  471
INFO:__main__:[Elapsed Time] : 7566.187143325806ms, 15.879765469977196ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8408,   396/  471
INFO:__main__:[Elapsed Time] : 7315.462350845337ms, 15.295034266532735ms on average

** --bert_model_name_or_path=./bert-checkpoint-nsmc-dha-cls
INFO:__main__:[Accuracy] : 0.6561,   309/  471
INFO:__main__:[Elapsed Time] : 7119.729280471802ms, 14.88446834239554ms on average

** --bert_model_name_or_path=./bert-checkpoint-nsmc-dha-cls --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8344,   393/  471
INFO:__main__:[Elapsed Time] : 6724.8547077178955ms, 14.073182166890895ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1
INFO:__main__:[Accuracy] : 0.6603,   311/  471
INFO:__main__:[Elapsed Time] : 15862.082242965698ms, 33.44054577198435ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1 --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8386,   395/  471
INFO:__main__:[Elapsed Time] : 15744.0345287323ms, 33.119978803269404ms on average

** --config=configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1 
INFO:__main__:[Accuracy] : 0.6454,   304/  471
INFO:__main__:[Elapsed Time] : 7038.804054260254ms, 14.745036591874792ms on average

** --config=configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1 --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8259,   389/  471
INFO:__main__:[Elapsed Time] : 6451.227426528931ms, 13.517273740565523ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/korean_hate_speech_morph --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
INFO:__main__:[Accuracy] : 0.6476,   305/  471
INFO:__main__:[Elapsed Time] : 6128.332614898682ms, 12.81673198050641ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8280,   390/  471
INFO:__main__:[Elapsed Time] : 6393.482446670532ms, 13.385195427752556ms on average

** --bert_model_name_or_path=./bert-checkpoint-nsmc-dha-cls
INFO:__main__:[Accuracy] : 0.6603,   311/  471
INFO:__main__:[Elapsed Time] : 6387.471914291382ms, 13.345901509548755ms on average

** --bert_model_name_or_path=./bert-checkpoint-nsmc-dha-cls --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8280,   390/  471
INFO:__main__:[Elapsed Time] : 6631.218910217285ms, 13.854946988694211ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1
INFO:__main__:[Accuracy] : 0.6667,   314/  471
INFO:__main__:[Elapsed Time] : 13503.794431686401ms, 28.387678430435507ms on average

** --bert_model_name_or_path=./embeddings/kor-bert-large-dha_bpe.v1 --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8386,   395/  471
INFO:__main__:[Elapsed Time] : 14023.700475692749ms, 29.498563928807034ms on average

** --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1 
INFO:__main__:[Accuracy] : 0.6242,   294/  471
INFO:__main__:[Elapsed Time] : 6239.235877990723ms, 13.059855014719862ms on average

** --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-dha-bert.v1 --data_dir=./data/korean_bias_speech_morph
INFO:__main__:[Accuracy] : 0.8323,   392/  471
INFO:__main__:[Elapsed Time] : 5894.242525100708ms, 12.312091665065035ms on average

```

</p>
</details>

#### BERT(kor-bert-base-bpe)

<details><summary><b>enc_class=cnn | cls</b></summary>
<p>

- train
```
* enc_class=cnn

$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --data_dir=./data/korean_hate_speech
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64  --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech --save_path=pytorch-model-kor-bert.pt

* enc_class=cls

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --data_dir=./data/korean_hate_speech
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/kor-bert-base-bpe.v1 --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64  --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech --save_path=pytorch-model-kor-bert.pt
```

- evaluation
```
* enc_class=cnn

$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/korean_hate_speech --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
INFO:__main__:[Accuracy] : 0.6327,   298/  471
INFO:__main__:[Elapsed Time] : 7187.519788742065ms, 15.07406234741211ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8280,   390/  471
INFO:__main__:[Elapsed Time] : 6787.185907363892ms, 14.202788535584794ms on average

** --config=configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 
INFO:__main__:[Accuracy] : 0.6093,   287/  471
INFO:__main__:[Elapsed Time] : 4213.6218547821045ms, 8.768345954570364ms on average

** --config=configs/config-distilbert-cnn.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8238,   388/  471
INFO:__main__:[Elapsed Time] : 4147.727012634277ms, 8.625224803356414ms on average

* enc_class=cls

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/korean_hate_speech --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
INFO:__main__:[Accuracy] : 0.6369,   300/  471
INFO:__main__:[Elapsed Time] : 6291.672468185425ms, 13.157690839564546ms on average

** --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8238,   388/  471
INFO:__main__:[Elapsed Time] : 6240.239858627319ms, 13.053022546971098ms on average

** --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 
INFO:__main__:[Accuracy] : 0.6136,   289/  471
INFO:__main__:[Elapsed Time] : 3709.501028060913ms, 7.698377649834815ms on average

** --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/kor-distil-bpe-bert.v1 --data_dir=./data/korean_bias_speech
INFO:__main__:[Accuracy] : 0.8153,   384/  471
INFO:__main__:[Elapsed Time] : 3626.2717247009277ms, 7.534836708231175ms on average

```

</p>
</details>



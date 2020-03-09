## 한국어 데이터 대상 실험

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

### BERT model

- 한국어 문서 데이터 준비
  - 다양한 문서 데이터를 크롤링

- [google original tf code](https://github.com/google-research/bert)를 이용해서 학습
  - [sentencepiece](https://github.com/google/sentencepiece) tokenizer 기반
    - ex) all.bpe.4.8m_step
  - `형태소분석기 tokenizer` 기반
    - ex) all.dha.2.5m_step

- huggingface 포맷으로 변환
  - [convert_bert_orignal_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/master/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py) 스크립트를 이용해서 변환.
  ```
  $ python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=all.bpe.4.8m_step/model.ckpt-4780000 --bert_config_file=all.bpe.4.8m_step/bert_config.json --pytorch_dump_path=pytorch_model.bin
  * 나머지 필요한 파일들은 huggingface에서 배포된 bert-base-cased에 있는 파일들을 복사해서 사용.
  * 단, vocab.txt는 tf에 있는 것을 그대로 활용.
  * config.json의 vocab_size 설정 필요.
  ```

### Glove model

- 한국어 문서 데이터 준비
  - 위 한국어 BERT 학습에 사용한 데이터를 그대로 이용(형태소분석기 tokenizer 사용).

- [Standford Glove code](https://github.com/stanfordnlp/GloVe)를 이용해서 한국어 Glove 학습
  - ex) kor.glove.300k.300d.txt

### Experiments summary

- iclassifier

|                     | Accuracy (%) |
| ------------------- | ------------ |
| Glove, CNN          | 87.31        |
| Glove, DenseNet-CNN | 88.18        |
| Glove, DenseNet-DSA | 87.66        |
| bpe BERT, CNN       | 89.45        |
| bpe BERT, CLS       | 89.31        |
| dha BERT, CNN       | **89.96**    |
| dha BERT, CLS       | 89.41        |

- [HanBert-nsmc](https://github.com/monologg/HanBert-nsmc#results)

|                   | Accuracy (%) |
| ----------------- | ------------ |
| HanBert-54kN      | 90.16        |
| HanBert-54kN-IP   | 88.72        |
| KoBERT            | 89.63        |
| DistilKoBERT      | 88.41        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

### Experiments with Glove

#### enc_class=cnn

- train
```
$ python preprocess.py --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --data_dir=data/clova_sentiments_morph --decay_rate=0.9 --embedding_trainable
```

- evaluation
```
$ python evaluate.py --data_dir=./data/clova_sentiments_morph 

INFO:__main__:[Accuracy] : 0.8731, 43653/49997
INFO:__main__:[Elapsed Time] : 97481ms, 1.9479358348667895ms on average
```

#### enc_class=densenet-cnn

- train
```
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --decay_rate=0.9

* iee_corpus_morph
$ python preprocess.py --config=configs/config-densenet-cnn-iee.json --data_dir=data/iee_corpus_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-cnn-iee.json --data_dir=data/iee_corpus_morph --decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/clova_sentiments_morph 

INFO:__main__:[Accuracy] : 0.8818, 44087/49997
INFO:__main__:[Elapsed Time] : 173152ms, 3.4614969197535803ms on average

* iee_corpus_morph
$ python evaluate.py --config=configs/config-densenet-cnn-iee.json --data_dir=./data/iee_corpus_morph 

```

#### enc_class=densenet-dsa

- train
```
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph --decay_rate=0.9

* iee_corpus_morph
$ python preprocess.py --config=configs/config-densenet-dsa-iee.json --data_dir=data/iee_corpus_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
$ python train.py --config=configs/config-densenet-dsa-iee.json --data_dir=data/iee_corpus_morph --decay_rate=0.9 --batch_size=256
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=./data/clova_sentiments_morph

INFO:__main__:[Accuracy] : 0.8766, 43827/49997
INFO:__main__:[Elapsed Time] : 348722ms, 6.973197855828467ms on average

* iee_corpus_morph
$ python evaluate.py --config=configs/config-densenet-dsa-iee.json --data_dir=./data/iee_corpus_morph
```

### Experiments with BERT(pytorch.all.bpe.4.8m_step)

- train
```
* enc_class=cnn
$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --data_dir=./data/clova_sentiments/

* enc_class=cls
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --data_dir=./data/clova_sentiments/
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8945, 44723/49997
INFO:__main__:[Elapsed Time] : 90526ms, 1.810628637718263ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8931, 44653/49997
INFO:__main__:[Elapsed Time] : 89785ms, 1.795807748464908ms on average
```

### Experiments with BERT(pytorch.all.dha.2.5m_step)
 
- train
```
* enc_class=cnn
$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/clova_sentiments_morph
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --data_dir=./data/clova_sentiments_morph/

* enc_class=cls
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=3 --data_dir=./data/clova_sentiments_morph/
```

- evaluation
```
* enc_class=cnn
$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=./data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8996, 44976/49997
INFO:__main__:[Elapsed Time] : 94477ms, 1.8896533792027521ms on average

* enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=./data/clova_sentiments_morph --bert_output_dir=bert-checkpoint

INFO:__main__:[Accuracy] : 0.8941, 44701/49997
INFO:__main__:[Elapsed Time] : 89692ms, 1.7939476368582115ms on average
```


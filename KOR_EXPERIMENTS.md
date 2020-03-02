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

|                   | Accuracy (%) |
| ----------------- | ----------- |
| bpe BERT, CNN     | 89.45       |
| bpe BERT, CLS     | 89.31       |
| dha BERT, CNN     | **89.96**   |
| dha BERT, CLS     | 89.41       |
| Glove, CNN        | 87.27       |

### Experiments with BERT(pytorch.all.bpe.4.8m_step)

- train
```
1) --bert_model_class=TextBertCNN
$ python preprocess.py --config=config-bert.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --data_dir=./data/clova_sentiments/ --batch_size=128

2) --bert_model_class=TextBertCLS
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --data_dir=./data/clova_sentiments/ --batch_size=128 --bert_model_class=TextBertCLS
```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments/test.txt.fs --label_path=data/clova_sentiments/label.txt --batch_size=128
INFO:__main__:[Accuracy] : 0.8945, 44723/49997
INFO:__main__:[Elapsed Time] : 90526ms, 1.810628637718263ms on average

2) --bert_model_class=TextBertCLS
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments/test.txt.fs --label_path=data/clova_sentiments/label.txt --batch_size=128 --bert_model_class=TextBertCLS --print_predicted_label > data/clova_sentiments/test.txt.predicted
INFO:__main__:[Accuracy] : 0.8931, 44653/49997
INFO:__main__:[Elapsed Time] : 89785ms, 1.795807748464908ms on average
$ paste data/clova_sentiments/test.txt data/clova_sentiments/test.txt.predicted | more
```

### Experiments with BERT(pytorch.all.dha.2.5m_step)
 
- train
```
1) --bert_model_class=TextBertCNN
$ python preprocess.py --config=config-bert.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/clova_sentiments_morph
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --data_dir=./data/clova_sentiments_morph/ --batch_size=128

2) --bert_model_class=TextBertCLS
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=3 --data_dir=./data/clova_sentiments_morph/ --batch_size=128 --bert_model_class=TextBertCLS
```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments_morph/test.txt.fs --label_path=data/clova_sentiments_morph/label.txt --batch_size=128 --print_predicted_label > data/clova_sentiments_morph/test.txt.predicted
INFO:__main__:[Accuracy] : 0.8996, 44976/49997
INFO:__main__:[Elapsed Time] : 94477ms, 1.8896533792027521ms on average
$ paste data/clova_sentiments_morph/test.txt data/clova_sentiments_morph/test.txt.predicted | more

2) --bert_model_class=TextBertCLS
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments_morph/test.txt.fs --label_path=data/clova_sentiments_morph/label.txt --batch_size=128 --bert_model_class=TextBertCLS --print_predicted_label > data/clova_sentiments_morph/test.txt.predicted
INFO:__main__:[Accuracy] : 0.8941, 44701/49997
INFO:__main__:[Elapsed Time] : 89692ms, 1.7939476368582115ms on average
```

### Experiments with Glove

- train
```
$ python preprocess.py --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
* embedding trainable
$ python train.py --data_dir=data/clova_sentiments_morph
```

- evaluation
```
$ python evaluate.py --data_path=data/clova_sentiments_morph/test.txt.ids --embedding_path=data/clova_sentiments_morph/embedding.npy --label_path=data/clova_sentiments_morph/label.txt
INFO:__main__:[Accuracy] : 0.8727, 43631/49997
INFO:__main__:[Elapsed Time] : 122452ms, 2.449186951217073ms on average
```

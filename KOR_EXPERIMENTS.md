## 한글 데이터 대상 실험 (1)

- data
  - [clova sentiment analysis data](https://github.com/e9t/nsmc)
    - set-up
      - 데이터를 다운받아서 './data/clova_sentiments/' 디레토리 아래, 'train.txt', 'valid.txt', 'test.txt' 생성.
      - 'test.txt'는 제공하지 않으므로 'valid.txt'를 복사해서 사용.
      - '*.txt' 데이터의 포맷은 './data/snips'의 데이터 포맷과 동일.
    - previous results
      - [SKT Brain에서 공개한 KoBERT를 적용한 성능](https://github.com/SKTBrain/KoBERT#naver-sentiment-analysis)
        - valid acc : 90.1%

- bert
  - 한글 문서 데이터를 [google original tf code](https://github.com/google-research/bert)로 학습한 결과물을 huggingface에서 제공하는 [convert_bert_orignal_tf_checkpoint_to_pytorch.py](https://github.com/huggingface/transformers/blob/master/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py) 스크립트를 이용해서 변환.
  ```
  $ python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=all.bpe.4.8m_step/model.ckpt-4780000 --bert_config_file=all.bpe.4.8m_step/bert_config.json --pytorch_dump_path=pytorch_model.bin
  * 나머지 필요한 파일들은 huggingface에서 배포된 bert-base-cased에 있는 파일들을 복사해서 사용.
  * 단, vocab.txt는 tf에 있는 것을 그대로 활용.
  * config.json의 vocab_size 설정 필요.
  ```
  - train
  ```
  1) --bert_model_class=TextBertCNN
  $ python preprocess.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
  $ python train.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=10 --data_dir=./data/clova_sentiments/ --batch_size=128
  ...
  1 epoch |  1172/ 1172 | train loss :  0.442, valid loss  0.423, valid acc 0.8851| lr :0.000020
  2 epoch |  1172/ 1172 | train loss :  0.401, valid loss  0.415, valid acc 0.8938| lr :0.000020
  3 epoch |  1172/ 1172 | train loss :  0.386, valid loss  0.415, valid acc 0.8945| lr :0.000020

  2) --bert_model_class=TextBertCLS
  $ python train.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=3 --data_dir=./data/clova_sentiments/ --batch_size=128 --bert_model_class=TextBertCLS
  ...
  1 epoch |  1172/ 1172 | train loss :  0.441, valid loss  0.419, valid acc 0.8887| lr :0.000020
  2 epoch |  1172/ 1172 | train loss :  0.401, valid loss  0.415, valid acc 0.8931| lr :0.000020
  3 epoch |  1172/ 1172 | train loss :  0.385, valid loss  0.417, valid acc 0.8913| lr :0.000020
  ```
  - evaluation
  ```
  1) --bert_model_class=TextBertCNN
  $ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments/test.txt.fs --label_path=data/clova_sentiments/label.txt --batch_size=128
  INFO:__main__:[Accuracy] : 0.8945, 44723/49997
  INFO:__main__:[Elapsed Time] : 90526ms, 1.810628637718263ms on average
  2) --bert_model_class=TextBertCLS
  $ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments/test.txt.fs --label_path=data/clova_sentiments/label.txt --batch_size=128 --bert_model_class=TextBertCLS
  INFO:__main__:[Accuracy] : 0.8931, 44653/49997
  INFO:__main__:[Elapsed Time] : 89785ms, 1.795807748464908ms on average
  ```

- glove
  - 위 한글 BERT 학습에 사용한 데이터를 그대로 이용해서 생성한 한글 glove 사용
  - 학습 데이터는 형태소분석기를 사용해서 tokenizing 필요
  - train
  ```
  $ python preprocess.py --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt
  $ python train.py --data_dir=data/clova_sentiments_morph
  ...
  5 epoch |  2344/ 2344 | train loss :  0.394, valid loss  0.439, valid acc 0.8676| lr :0.000250
  ```
  - evaluation
  ```
  $ python evaluate.py --data_path=data/clova_sentiments_morph/test.txt.ids --embedding_path=data/clova_sentiments_morph/embedding.npy --label_path=data/clova_sentiments/label.txt
  INFO:__main__:[Accuracy] : 0.8676, 43377/49997
  INFO:__main__:[Elapsed Time] : 78819ms, 1.5764745884753084ms on average
  ```

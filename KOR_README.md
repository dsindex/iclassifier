## Clova sentiment classification 데이터를 대상으로 실험
  - BERT
    - 한글 문서 데이터에 대해 google original tf code로 학습한 결과물을 huggingface에서 제공하는 convert_bert_orignal_tf_checkpoint_to_pytorch.py 스크립트를 이용해서 변환
    ```
    $ python convert_bert_original_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=all.bpe.4.8m_step/model.ckpt-4780000 --bert_config_file=all.bpe.4.8m_step/bert_config.json --pytorch_dump_path=pytorch_model.bin
    * 나머지 필요한 파일들은 huggingface에서 배포된 bert-base-cased에 있는 파일들을 복사해서 사용. 단, vocab.txt는 tf에 있는 것을 그대로 활용. config.json의 vocab_size 설정 필요.
    ```
  - train
  ```
  $ python preprocess.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
  $ python train.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=3e-5 --epoch=3 --data_dir=./data/clova_sentiments/ --batch_size=128 --bert_model_class=TextBertCLS
  * bert_model_class를 TextBertCNN으로 하는 경우, 성능이 낮게 나오는 문제가 있음.
  ...
  1 epoch |  1172/ 1172 | train loss :  1.270, valid loss  1.253, valid acc 0.8758| lr :0.000030
  2 epoch |  1172/ 1172 | train loss :  1.232, valid loss  1.242, valid acc 0.8898| lr :0.000030
  3 epoch |  1172/ 1172 | train loss :  1.222, valid loss  1.247, valid acc 0.8880| lr :0.000030
  ```
  - evaluation
  ```
  $ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --data_path=data/clova_sentiments/test.txt.fs --batch_size=128 --bert_model_class=TextBertCLS
  INFO:__main__:[Accuracy] : 0.8898, 44487/49997
  INFO:__main__:[Elapsed Time] : 89514ms, 1.7903874232453947ms on average
  ```

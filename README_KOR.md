1. Clova sentiment classification 데이터를 대상으로 실험
  - train
  ```
  $ python preprocess.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step --data_dir=./data/clova_sentiments
  $ python train.py --emb_class=bert --bert_model_name_or_path=./pytorch.all.bpe.4.8m_step/ --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --data_dir=./data/clova_sentiments/ --batch_size=128 --bert_model_class=TextBertCLS
  ```

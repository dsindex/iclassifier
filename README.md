## iclassifier

reference pytorch code for intent(sentence) classification

## requirements

- python >= 3.6

- pip install -r requirements.txt

- pretrained embedding
  - glove
    - [download Glove6B](http://nlp.stanford.edu/data/glove.6B.zip)
  - unzip to 'embeddings' dir
  ```
  $ mkdir embeddings
  $ ls embeddings
  glove.6B.zip
  $ unzip glove.6B.zip 
  ```

- additional requirements for BERT(huggingface's [transformers](https://github.com/huggingface/transformers.git))
```
$ pip install tensorflow-gpu==2.0
$ pip install git+https://github.com/huggingface/transformers.git
```

- data
  - Snips
    - `data/snips`
    - from [joint-intent-classification-and-slot-filling-based-on-BERT](https://github.com/lytum/joint-intent-classification-and-slot-filling-based-on-BERT)
    - paper : [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf)
      - intent classification accuracy : **98.6%** (test set)
    - [previous SOTA on SNIPS data](https://paperswithcode.com/sota/intent-detection-on-snips)
      - intent classification accuracy : 97.7% (test set)
  - SST-2
    - `data/sst2`
    - from [GLUE benchmark data](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py)
      - `test.txt` from [pytorch-sentiment-classification](https://github.com/clairett/pytorch-sentiment-classification)
    - [SOTA on SST2 data](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary)
      - sentence classification accuracy : **97.4%** (valid set)
      - [GLUE leaderboard](https://gluebenchmark.com/leaderboard/)
  - TCCC
    - [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview)

## Snips data

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py
[Accuracy] : 0.9771428571428571, 684/700
[Elapsed Time] : 1327ms, 1.8957142857142857ms on average
```

- best : **97.71%** (test set)

### emb_class=bert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
$ python preprocess.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case

* fine-tuning
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --bert_model_class=TextBertCLS

* feature-based
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based --bert_model_class=TextBertCLS

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs

* fine-tuning
INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 9353ms, 13.361428571428572ms on average
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9800,   686/  700
  INFO:__main__:[Elapsed Time] : 16994ms, 24.277142857142856ms on average

* feature-based, --epoch=30
INFO:__main__:[Accuracy] : 0.9628571428571429, 674/700
INFO:__main__:[Elapsed Time] : 11480ms, 16.4ms on average

2) --bert_model_class=TextBertCLS
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs --bert_model_class=TextBertCLS

* fine-tuning
INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 8940ms, 12.771428571428572ms on average
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9786,   685/  700
  INFO:__main__:[Elapsed Time] : 16480ms, 23.542857142857144ms on average

* feature-based, --epoch=100
INFO:__main__:[Accuracy] : 0.8871428571428571, 621/700
INFO:__main__:[Elapsed Time] : 11323ms, 16.175714285714285ms on average
```

- best : **98.00%** (test set)

## SST-2 data

### emb_class=glove

- train
```
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/sst2
$ python train.py --data_dir=data/sst2 --lr=0.001
```

- evaluation
```
$ python evaluate.py --data_path=data/sst2/test.txt.ids --embedding_path=data/sst2/embedding.npy --label_path=data/sst2/label.txt
INFO:__main__:[Accuracy] : 0.8155,  1485/ 1821
INFO:__main__:[Elapsed Time] : 2908ms, 1.5969247666117519ms on average
```

- best : **81.55%** (test set)

### emb_class=bert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
$ python preprocess.py --emb_class=bert --data_dir=data/sst2 --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case

* fine-tuning
$ python train.py --emb_class=bert --data_dir=data/sst2 --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3
$ python train.py --emb_class=bert --data_dir=data/sst2 --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --bert_model_class=TextBertCLS

```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/sst2/test.txt.fs --label_path=data/sst2/label.txt

* fine-tuning
INFO:__main__:[Accuracy] : 0.9143,  1665/ 1821
INFO:__main__:[Elapsed Time] : 25373ms, 13.933552992861065ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9253,  1685/ 1821
  INFO:__main__:[Elapsed Time] : 55444ms, 30.44700713893465ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=1e-5
  INFO:__main__:[Accuracy] : 0.9308,  1695/ 1821
  INFO:__main__:[Elapsed Time] : 52170ms, 28.649093904448105ms on average

2) --bert_model_class=TextBertCLS
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/sst2/test.txt.fs --label_path=data/sst2/label.txt --bert_model_class=TextBertCLS

* fine-tuning
INFO:__main__:[Accuracy] : 0.8929,  1626/ 1821
INFO:__main__:[Elapsed Time] : 23413ms, 12.85722130697419ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9385,  1709/ 1821
  INFO:__main__:[Elapsed Time] : 50982ms, 27.99670510708402ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=1e-5
  INFO:__main__:[Accuracy] : 0.9292,  1692/ 1821
  INFO:__main__:[Elapsed Time] : 48522ms, 26.645799011532127ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5 --batch_size=32
  INFO:__main__:[Accuracy] : 0.9116,  1660/ 1821
  INFO:__main__:[Elapsed Time] : 45190ms, 24.816035145524438ms on average
```

- best : **93.85%** (test set)

### emb_class=albert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
$ python preprocess.py --emb_class=albert --data_dir=data/sst2 --bert_model_name_or_path=./albert-xlarge-v2

* feature-based
$ python train.py --emb_class=albert --data_dir=data/sst2 --bert_model_name_or_path=./albert-xlarge-v2 --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=5 --bert_use_feature_based

```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --emb_class=albert --bert_output_dir=bert-checkpoint --data_path=data/sst2/test.txt.fs --label_path=data/sst2/label.txt
INFO:__main__:[Accuracy] : 0.8429,  1535/ 1821
INFO:__main__:[Elapsed Time] : 60409ms, 33.17353102690829ms on average
```

- best : **84.29%** (test set)

## experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

## references

- [Intent Detection](https://paperswithcode.com/task/intent-detection)
- [Intent Classification](https://paperswithcode.com/task/intent-classification)
- [A Visual Guide to Using BERT for the First Time](https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)
- [Identifying Hate Speech with BERT and CNN](https://towardsdatascience.com/identifying-hate-speech-with-bert-and-cnn-b7aa2cddd60d)


## iclassifier

reference pytorch code for intent(sentence) classification.
- embedding
  - Glove, BERT, ALBERT
- encoding
  - CNN
  - DenseNet
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
    - implementation from [ntagger](https://github.com/dsindex/ntagger/blob/master/model.py#L43)
- decoding
  - Softmax

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
  - BERT(huggingface's [transformers](https://github.com/huggingface/transformers.git))
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

### experiments summary

|                   | Accuracy (%)|
| ----------------- | ----------- |
| Glove, CNN        | 97.71       |
| BERT, CNN         | **98.00**   |
| BERT, CLS         | 97.86       |

### emb_class=glove

- train
```
* token_emb_dim in config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
* embedding trainable
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

### emb_class=bert

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=config-bert.json --emb_class=bert --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case

* fine-tuning
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --bert_model_class=TextBertCLS

* feature-based
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based
$ python train.py --config=config-bert.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based --bert_model_class=TextBertCLS

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs

* fine-tuning
INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 9353ms, 13.361428571428572ms on average
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9800,   686/  700
  INFO:__main__:[Elapsed Time] : 16994ms, 24.277142857142856ms on average

2) --bert_model_class=TextBertCLS
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs --bert_model_class=TextBertCLS

* fine-tuning
INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 8940ms, 12.771428571428572ms on average
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9786,   685/  700
  INFO:__main__:[Elapsed Time] : 16480ms, 23.542857142857144ms on average
```

## SST-2 data

### experiments summary

|                   | Accuracy (%)|
| ----------------- | ----------- |
| Glove, CNN        | 83.64       |
| BERT, CNN         | 93.08       |
| BERT, CLS         | **93.85**   |
| ALBERT, CNN       | 86.66       |

### emb_class=glove

- train
```
* token_emb_dim in config-glove.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/sst2
* embedding trainable
$ python train.py --data_dir=data/sst2
```

- evaluation
```
$ python evaluate.py --data_path=data/sst2/test.txt.ids --embedding_path=data/sst2/embedding.npy --label_path=data/sst2/label.txt
INFO:__main__:[Accuracy] : 0.8248,  1502/ 1821
INFO:__main__:[Elapsed Time] : 4627ms, 2.540911587040088ms on average
  * single layer fc, no layernorm 
  INFO:__main__:[Accuracy] : 0.8364,  1523/ 1821
  INFO:__main__:[Elapsed Time] : 4300ms, 2.361339923119165ms on average
```

### emb_class=bert

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=config-bert.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case

* fine-tuning
$ python train.py --config=config-bert.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3
$ python train.py --config=config-bert.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --bert_model_class=TextBertCLS

```

- evaluation
```
1) --bert_model_class=TextBertCNN
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/sst2/test.txt.fs --label_path=data/sst2/label.txt

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
$ python evaluate.py --config=config-bert.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/sst2/test.txt.fs --label_path=data/sst2/label.txt --bert_model_class=TextBertCLS

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

### emb_class=albert

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=config-albert.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2

* feature-based
$ python train.py --config=config-albert.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2 --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=15 --bert_use_feature_based

```

- evaluation
```
1) --bert_model_class=TextBertCNN
  * albert-base-v2
  $ python evaluate.py --config=config-albert.json --bert_output_dir=bert-checkpoint --data_path=data/sst2/test.txt.fs --label_path=data/sst2/label.txt
  INFO:__main__:[Accuracy] : 0.8666,  1578/ 1821
  INFO:__main__:[Elapsed Time] : 30896ms, 16.966501922020868ms on average
  
```

## experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

## references

- [Intent Detection](https://paperswithcode.com/task/intent-detection)
- [Intent Classification](https://paperswithcode.com/task/intent-classification)
- [Identifying Hate Speech with BERT and CNN](https://towardsdatascience.com/identifying-hate-speech-with-bert-and-cnn-b7aa2cddd60d)


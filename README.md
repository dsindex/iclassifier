## iclassifier

reference pytorch code for intent(sentence) classification.
- embedding
  - Glove, BERT, ALBERT, ROBERTa
- encoding
  - CNN
  - DenseNet
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
    - implementation from [ntagger](https://github.com/dsindex/ntagger)
  - DSA(Dynamic Self Attention)
    - [Dynamic Self-Attention: Computing Attention over Words Dynamically for Sentence Embedding](https://arxiv.org/pdf/1808.07383.pdf)
  - CLS
    - classified by '[CLS]' only for BERT-like architectures. 
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
  $ pip install tensorflow-gpu==2.0.0
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

|                     | Accuracy (%)|
| ------------------- | ----------- |
| Glove, CNN          | 97.86       |
| Glove, Densenet-CNN | 97.57       |
| Glove, Densenet-DSA | 97.43       |
| BERT-large, CNN     | **98.00**   |
| BERT-large, CLS     | 97.86       |

### emb_class=glove, enc_class=cnn

- train
```
* token_emb_dim in configs/config-glove-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py --decay_rate=0.9 --embedding_trainable

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py
INFO:__main__:[Accuracy] : 0.9786,   685/  700
INFO:__main__:[Elapsed Time] : 1351ms, 1.793991416309013ms on average
```

### emb_class=glove, enc_class=densenet-cnn

- train
```
* token_emb_dim in configs/config-densenet-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-cnn.json
$ python train.py --config=configs/config-densenet-cnn.json --decay_rate=0.9 --embedding_trainable
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json

INFO:__main__:[Accuracy] : 0.9757,   683/  700
INFO:__main__:[Elapsed Time] : 2633ms, 3.609442060085837ms on average
```

### emb_class=glove, enc_class=densenet-dsa

- train
```
* token_emb_dim in configs/config-densenet-dsa.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-dsa.json
$ python train.py --config=configs/config-densenet-dsa.json --decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 5367ms, 7.500715307582261ms on average
```

### emb_class=bert, enc_class=cnn | cls

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

* --bert_use_feature_based for feature-based
```

- evaluation
```
1) enc_class=cnn
$ python evaluate.py --config=configs/config-bert-cnn.json --bert_output_dir=bert-checkpoint --bert_do_lower_case

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 9353ms, 13.361428571428572ms on average
  
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9800,   686/  700
  INFO:__main__:[Elapsed Time] : 16994ms, 24.277142857142856ms on average

2) enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --bert_do_lower_case

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 8940ms, 12.771428571428572ms on average
  
  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9786,   685/  700
  INFO:__main__:[Elapsed Time] : 16480ms, 23.542857142857144ms on average
```

## SST-2 data

### experiments summary

- iclassifier

|                     | Accuracy (%)| Etc           |
| ------------------- | ----------- | ------------- |
| Glove, CNN          | 83.42       |               |
| Glove, DenseNet-CNN | 86.38       |               |
| Glove, DenseNet-DSA | 85.34       |               |
| BERT-large, CNN     | 93.08       |               |
| BERT-large, CLS     | **93.85**   |               |
| ALBERT-base, CNN    | 86.66       | feature-based |             
| ALBERT-xxlarge, CNN | 91.32       | feature-based |
| ROBERTa-large, CNN  | -           |               |
| ROBERTa-large, CLS  | -           |               |

- [sst2 learderboard](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary)

|                   | Accuracy (%)|
| ----------------- | ----------- |
| T5-3B             | 97.4        |
| ALBERT            | 97.1        |
| RoBERTa           | 96.7        |
| MT-DNN            | 95.6        |
| DistilBERT        | 92.7        |

### emb_class=glove, enc_class=cnn

- train
```
* token_emb_dim in configs/config-glove-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --data_dir=data/sst2
$ python train.py --data_dir=data/sst2 --decay_rate=0.9 
```

- evaluation
```
$ python evaluate.py --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8342,  1519/ 1821
INFO:__main__:[Elapsed Time] : 3161ms, 1.6873626373626374ms on average
```

### emb_class=glove, enc_class=densenet-cnn

- train
```
* token_emb_dim in configs/config-densenet-cnn.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2
$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8638,  1573/ 1821
INFO:__main__:[Elapsed Time] : 6678ms, 3.6203296703296703ms on average
```

### emb_class=glove, enc_class=densenet-dsa

- train
```
* token_emb_dim in configs/config-densenet-dsa.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2
$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --decay_rate=0.9
```

- evaluation
```
$ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2

INFO:__main__:[Accuracy] : 0.8534,  1554/ 1821
INFO:__main__:[Elapsed Time] : 11459ms, 6.245054945054945ms on average
```

### emb_class=bert, enc_class=cnn | cls

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64
```

- evaluation
```
1) enc_class=cnn
$ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --bert_do_lower_case 

INFO:__main__:[Accuracy] : 0.9143,  1665/ 1821
INFO:__main__:[Elapsed Time] : 25373ms, 13.933552992861065ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=2e-5
  INFO:__main__:[Accuracy] : 0.9253,  1685/ 1821
  INFO:__main__:[Elapsed Time] : 55444ms, 30.44700713893465ms on average

  ** --bert_model_name_or_path=bert-large-uncased --lr=1e-5
  INFO:__main__:[Accuracy] : 0.9308,  1695/ 1821
  INFO:__main__:[Elapsed Time] : 52170ms, 28.649093904448105ms on average

2) enc_class=cls
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --bert_do_lower_case

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

### emb_class=albert, enc_class=cnn | cls

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2

* feature-based
$ python train.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-base-v2 --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=15 --batch_size=64 --bert_use_feature_based

* albert-xxlarge-v2
$ python preprocess.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-xxlarge-v2
$ python train.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/albert-xxlarge-v2 --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=15 --batch_size=32 --bert_use_feature_based

```

- evaluation
```
$ python evaluate.py --config=configs/config-albert-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint 

* feature-based
* fine-tuning ALBERT doesn't work well. i guess ALBERT needs more data.
INFO:__main__:[Accuracy] : 0.8666,  1578/ 1821
INFO:__main__:[Elapsed Time] : 30896ms, 16.966501922020868ms on average
 
* albert-xxlarge-v2
INFO:__main__:[Accuracy] : 0.9132,  1663/ 1821
INFO:__main__:[Elapsed Time] : 102140ms, 56.090060406370127ms on average
```

### emb_class=roberta, enc_class=cnn | cls

- train
```
* n_ctx size should be less than 512
$ python preprocess.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large
$ python train.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large 
$ python train.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64
```

- evaluation
```
1) enc_class=cnn
$ python evaluate.py --config=configs/config-roberta-cnn.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint


2) enc_class=cls
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint

```

## experiments for Korean

- [KOR_EXPERIMENTS.md](/KOR_EXPERIMENTS.md)

## references

- [Intent Detection](https://paperswithcode.com/task/intent-detection)
- [Intent Classification](https://paperswithcode.com/task/intent-classification)
- [Identifying Hate Speech with BERT and CNN](https://towardsdatascience.com/identifying-hate-speech-with-bert-and-cnn-b7aa2cddd60d)


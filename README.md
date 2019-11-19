## iclassifier

reference pytorch code for intent classification

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

- data
  - snips
    - from https://github.com/lytum/joint-intent-classification-and-slot-filling-based-on-BERT
    - paper : https://arxiv.org/pdf/1902.10909.pdf

- additional requirements for BERT(huggingface's transformers)
```
$ pip install tensorflow-gpu==2.0
$ pip install git+https://github.com/huggingface/transformers.git
```

## usage

- train
```
$ rm -rf runs bert-checkpoint

1. emb_class=glove
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py

2. emb_class=bert
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
$ python preprocess.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint

* tensorboardX
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
1. emb_class=glove
$ python evaluate.py
...
[Accuracy] : 0.9771428571428571, 684/700
[Elapsed Time] : 1327ms, 1.8957142857142857ms on average

2. emb_class=bert
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs
...
INFO:__main__:[Accuracy] : 0.9942857142857143, 696/700
INFO:__main__:[Elapsed Time] : 10691ms, 15.272857142857143ms on average
```




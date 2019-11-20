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
    - from [joint-intent-classification-and-slot-filling-based-on-BERT](https://github.com/lytum/joint-intent-classification-and-slot-filling-based-on-BERT)
    - paper : [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf)
    - intent classification accuracy : 98.6%
    - [previous SOTA on SNIPS data](https://paperswithcode.com/sota/intent-detection-on-snips)


- additional requirements for BERT(huggingface's transformers)
```
$ pip install tensorflow-gpu==2.0
$ pip install git+https://github.com/huggingface/transformers.git
```

## usage, emb_class=glove

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

## usage, emb_class=bert

- train
```
* ignore token_emb_dim in config.json
* n_ctx size should be less than 512
$ python preprocess.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case

* fine-tuning
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3

* feature-based
$ python train.py --emb_class=bert --bert_model_name_or_path=bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --bert_use_feature_based

* default model class is TextBertCNN, possible to set --bert_model_class=TextBertCLS

* tensorboardX
$ rm -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py --emb_class=bert --bert_output_dir=bert-checkpoint --bert_do_lower_case --data_path=data/snips/test.txt.fs

1) --bert_model_class=TextBertCNN

* fine-tuning
INFO:__main__:[Accuracy] : 0.9785714285714285, 685/700
INFO:__main__:[Elapsed Time] : 11805ms, 16.864285714285714ms on average
~ 
INFO:__main__:[Accuracy] : 0.9828571428571429, 688/700
INFO:__main__:[Elapsed Time] : 10772ms, 15.388571428571428ms on average

* feature-based, --epoch=30
INFO:__main__:[Accuracy] : 0.9628571428571429, 674/700
INFO:__main__:[Elapsed Time] : 11480ms, 16.4ms on average

2) --bert_model_class=TextBertCLS

* fine-tuning
INFO:__main__:[Accuracy] : 0.9785714285714285, 685/700
INFO:__main__:[Elapsed Time] : 11864ms, 16.94857142857143ms on average

* feature-based, --epoch=100
INFO:__main__:[Accuracy] : 0.8871428571428571, 621/700
INFO:__main__:[Elapsed Time] : 11323ms, 16.175714285714285ms on average
```

## references

- [Intent Detection](https://paperswithcode.com/task/intent-detection)
- [Intent Classification](https://paperswithcode.com/task/intent-classification)


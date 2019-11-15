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

## usage

- train
```
* token_emb_dim in config.json == 300 (ex, glove.6B.300d.txt )
$ python preprocess.py
$ python train.py

* tensorboardX
$ run -rf runs
$ tensorboard --logdir runs/ --port port-number --bind_all
```

- evaluation
```
$ python evaluate.py
[Loading model...]
[Loaded]
[Test data loaded]
[Accuracy] : 0.9771428571428571, 684/700
[Elapsed Time] : 1327ms, 1.8957142857142857ms on average
```




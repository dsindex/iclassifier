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

## usage

- train
```
* train, valid data
$ python preprocess.py
$ python train.py
```

- evaluation
```
* test data
$ python evaluate.py
...
[Accuracy] : 0.97, 679/700
[Elapsed Time] : 1197ms, 1.71ms on average
```




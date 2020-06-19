#### Description

Distilling BERT(RoBERTa, ELECTRA) based model to GloVe based small model

#### Prerequisites
```
$ python -m pip install spacy
$ python -m spacy download en_core_web_sm
```

#### Train teacher model

- BERT-large, CLS (bert-large-uncased)
```
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-large-uncased --bert_do_lower_case
$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-large-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9412,  1714/ 1821
INFO:__main__:[Elapsed Time] : 40847.62740135193ms, 22.37672412788475ms on average
```

- RoBERTa-large, CLS(roberta-large)
```
$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large 
$ python train.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --lr_decay_rate=0.9 --batch_size=64
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9325,  1698/ 1821
INFO:__main__:[Elapsed Time] : 45188.396692276ms, 24.755752610636282ms on average

$ python train.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --batch_size=64


```

- ELECTRA-large, CLS (electra-large-discriminator)
```
$ python preprocess.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_do_lower_case
$ python train.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_output_dir=bert-checkpoint --lr=1e-6 --epoch=15 --lr_decay_rate=0.9 --batch_size=64 --bert_do_lower_case
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9643,  1756/ 1821
INFO:__main__:[Elapsed Time] : 41302.36577987671ms, 22.629007140358727ms on average
```

#### Generate pseudo labeled data

- augmentation
```
* bert, electra
$ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw

* roberta
$ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw --mask_token='<mask>'
```

- add logits by teacher model
```
* converting augmented.raw to augmented.raw.fs(id mapped file)
* labeling augmented.raw to augmented.raw.pred

* bert
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-large-uncased --bert_do_lower_case --augmented
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

* roberta
$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --augmented
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

* electra 
$ python preprocess.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_do_lower_case --augmented
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

$ cp data/sst2/augmented.raw.pred data/sst2/augmented.txt
```

#### Train student model

- Glove, CNN
  - from bert
  ```
  * converting augmented.txt to augmented.txt.ids(id mapped file) and train!
  $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --augmented
  $ python train.py --config=configs/config-glove-cnn.json --data_dir=data/sst2 --lr=1e-3 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --save_path=pytorch-model-cnn.pt --augmented
  $ python evaluate.py --config=configs/config-glove-cnn.json --data_dir=data/sst2 --model_path=pytorch-model-cnn.pt
  INFO:__main__:[Accuracy] : 0.8616,  1569/ 1821
  INFO:__main__:[Elapsed Time] : 3341.681718826294ms, 1.7900076541271839ms on average
  ```
  - from roberta
  ```

  ```
  - from electra
  ```
  INFO:__main__:[Accuracy] : 0.8655,  1576/ 1821
  INFO:__main__:[Elapsed Time] : 3255.631446838379ms, 1.7466542484996084ms on average
  ```

- GloVe, DenseNet-CNN
  - from bert
  ```
  $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --augmented
  $ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --save_path=pytorch-model-densenet.pt --augmented
  $ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --model_path=pytorch-model-densenet.pt
  INFO:__main__:[Accuracy] : 0.8852,  1612/ 1821
  INFO:__main__:[Elapsed Time] : 6774.356126785278ms, 3.678809417473091ms on average

  ```
  - from roberta
  ```

  ```
  - from electra
  ```
  INFO:__main__:[Accuracy] : 0.8979,  1635/ 1821
  INFO:__main__:[Elapsed Time] : 6723.0706214904785ms, 3.640611617119758ms on average
  ```

- Glove, DenseNet-DSA
  - from bert
  ```
  $ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --augmented
  $ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --save_path=pytorch-model-densenet.pt --augmented
  $ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --model_path=pytorch-model-densenet.pt
  INFO:__main__:[Accuracy] : 0.8814,  1605/ 1821
  INFO:__main__:[Elapsed Time] : 15502.179622650146ms, 8.464712756020683ms on average
  ```
  - from roberta
  ```

  ```
  - from electra
  ```
  INFO:__main__:[Accuracy] : 0.8858,  1613/ 1821
  INFO:__main__:[Elapsed Time] : 15340.755224227905ms, 8.370806751670418ms on average
  ```

#### Experiments for NSMC corpus

```

```


#### References

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
  - [distil-bilstm](https://github.com/dsindex/distil-bilstm)

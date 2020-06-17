### Distilling BERT(ELECTRA) based model to GloVe based small model

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
```

- ELECTRA-large, CLS (electra-large-discriminator)
```
$ python preprocess.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_do_lower_case
$ python train.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_output_dir=bert-checkpoint --lr=1e-6 --epoch=15 --lr_decay_rate=0.9 --batch_size=64 --bert_do_lower_case
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
```

#### Generate pseudo labeled data

- augmentation
```
$ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw
```

- add logits by teacher model
```
* pseudo labeling augmented.raw

* bert
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --enable_inference --augmented

* electra 
$ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --enable_inference --augmented

$ cp data/sst2/augmented.raw.inference data/sst2/augmented.txt
```

#### Train student model

- GloVe, DenseNet-CNN
  - from bert
  ```
  $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --augmented
  $ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --lr_decay_rate=0.9 --save_path=pytorch-model-densenet.pt
  $ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --model_path=pytorch-model-densenet.pt
  INFO:__main__:[Accuracy] : 0.8847,  1611/ 1821
  INFO:__main__:[Elapsed Time] : 6953.445672988892ms, 3.77188632776449ms on average
  ```
  - from electra
  ```

  ```

#### Experiments for NSMC corpus

```

```


#### References

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
  - [distil-bilstm](https://github.com/dsindex/distil-bilstm)

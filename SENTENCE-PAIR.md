## Sentence Pair Classification

#### KorNLI

- data preparation
```
$ cd data
$ git clone https://github.com/kakaobrain/KorNLUDatasets.git
$ mkdir kor_nli
$ cat KorNLUDatasets/KorNLI/multinli* KorNLUDatasets/KorNLI/snli* > kor_nli/train.txt
$ cat KorNLUDatasets/xnli.dev* > kor_nli/valid.txt
$ cat KorNLUDatasets/xnli.test* > kor_nli/test.txt
```

- train
```
$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base --data_dir=./data/kor_nli

$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/kor_nli

```

- evaluate
```
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/kor_nli --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.8286,  4152/ 5011
INFO:__main__:[Elapsed Time] : 65109.9374294281ms, 12.96194004203507ms on average

cp -rf bert-checkpoint embeddings/klue-roberta-base-kornli
```

#### apply to NSMC

- FULL
```
$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli --data_dir=./data/clova_sentiments
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments

$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.9049, 45240/49997
INFO:__main__:[Elapsed Time] : 646106.6389083862ms, 12.919756362644021ms on average

# klue-roberta-base 그대로 fine-tuning하면 91.18%가 나오는데, 성능 하락이 있음.
# 학습을 few-shot으로 하면 상황이 다를까?

```

- FEW-SHOT
```


```

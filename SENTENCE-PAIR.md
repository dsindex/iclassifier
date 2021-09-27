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

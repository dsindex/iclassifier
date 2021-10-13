# Sentence Pair Classification 

#### MNLI

- data preparation
```
`data/mnli` from [GLUE benchmark data](https://github.com/nyu-mll/GLUE-baselines/blob/master/download_glue_data.py)

$ cd data/mnli
$ python extract.py --input_path train.tsv > train.txt
$ python extract.py --input_path dev_matched.tsv > valid_matched.txt
$ python extract.py --input_path dev_mismatched.tsv > valid_mismatched.txt
$ cat valid_matched.txt valid_mismatched.txt > valid.txt
$ cp valid.txt test.txt
```

- train
```
$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/roberta-base --data_dir=./data/mnli
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/roberta-base --bert_output_dir=bert-checkpoint-mnli --save_path=pytorch-model-mnli.pt --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/mnli --eval_steps=-1
```

- evaluate
```
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/mnli --model_path=pytorch-model-mnli.pt --bert_output_dir=bert-checkpoint-mnli
INFO:__main__:[Accuracy] : 0.8757, 17204/19647
INFO:__main__:[Elapsed Time] : 215832.4155807495ms, 10.962099246479738ms on average
```

- adversarial test
```
`data/adv_glue` from https://adversarialglue.github.io/dataset/dev.zip

$ cd data/adv_glue
$ python extract.py --input_path dev.json --dataset mnli > mnli.txt

* preprocessing
$ cd ../..
$ cp -rf data/mnli/train.txt data/adv_glue/train.txt
$ cp -rf data/mnli/valid.txt data/adv_glue/valid.txt
$ cp -rf data/adv_glue/mnli.txt data/adv_glue/test.txt
$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/roberta-base --data_dir=./data/adv_glue

* roberta-base
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/adv_glue --model_path=pytorch-model-mnli.pt --bert_output_dir=bert-checkpoint-mnli 
INFO:__main__:[Accuracy] : 0.1653,    20/  121
INFO:__main__:[Elapsed Time] : 1855.2687168121338ms, 11.679033438364664ms on average

** --use_isomax --criterion=IsoMaxLoss
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/roberta-base --bert_output_dir=bert-checkpoint-mnli --save_path=pytorch-model-mnli.pt --lr=2e-5 --epoch=5 --batch_size=64 --data_dir=./data/mnli --eval_steps=-1 --use_isomax --criterion=IsoMaxLoss


```


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



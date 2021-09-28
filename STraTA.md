# Intermediate fine-tuning and self-training


## Reference

- [STraTA: Self-Training with Task Augmentation for Better Few-shot Learning](https://arxiv.org/pdf/2109.06270.pdf)


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


## Few-Shot Learning

#### NSMC

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
# 학습 데이터 20, 개발 데이터 256, 평가 데이터 동일
$ wc -l train.txt valid.txt test.txt unlabeled.txt
      20 train.txt
     256 valid.txt
   49997 test.txt
  149719 unlabeled.txt
  199992 total
# NLI를 이용한 intermediate finetuning 방법이 효과적인지 실험

# klue-roberta-base
$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base --data_dir=./data/clova_sentiments_fewshot
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments_fewshot

$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.5438, 27186/49997
INFO:__main__:[Elapsed Time] : 654522.3729610443ms, 13.087259439365685ms on average

# klue-roberta-base-kornli
$ python preprocess.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli --data_dir=./data/clova_sentiments_fewshot
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments_fewshot

$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.5085, 25422/49997
INFO:__main__:[Elapsed Time] : 656467.6933288574ms, 13.127830321526124ms on average

# STraTA 논문과는 다르게 오히려 성능 하락이 있음.
  GLUE MNLI도 bert-base로 학습시 83%, bert-large로 학습시 86% 정도인데, 이를 이용해서 SST-2에 효과가 있었다면
  비슷하게 KorNLI에서 학습한 것도 NSMC에 효과가 있어야 할것 같은데...
  batch size와 learning rate가 적절하지 않아 보이는데, optuna를 사용해본다.
  train.py 수정 필요.
    bsz = trial.suggest_categorical('batch_size', [1, 32]) )

# klue-roberta-base
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base --bert_output_dir=bert-checkpoint --data_dir=./data/clova_sentiments_fewshot --hp_search_optuna --hp_trials=24 --epoch=12 --patience=4
INFO:__main__:[study.best_params] : {'lr': 1.8124305795132615e-05, 'batch_size': 1, 'seed': 24, 'epochs': 5}
INFO:__main__:[study.best_value] : 0.55078125

$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base --bert_output_dir=bert-checkpoint --data_dir=./data/clova_sentiments_fewshot --lr=1.8124305795132615e-05 --batch_size=1 --seed=24 --epoch=20
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.6674, 33366/49997
INFO:__main__:[Elapsed Time] : 651090.7757282257ms, 13.0190913968491ms on average

# klue-roberta-base-kornli
$ python train.py --config=configs/config-roberta-cls.json --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli --bert_output_dir=bert-checkpoint --data_dir=./data/clova_sentiments_fewshot --lr=1.8124305795132615e-05 --batch_size=1 --seed=24 --epoch=20

$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_output_dir=bert-checkpoint
INFO:__main__:[Accuracy] : 0.6972, 34859/49997
INFO:__main__:[Elapsed Time] : 652646.0647583008ms, 13.051110626707192ms on average

# 약 3% 정도 성능 향상이 보임.

$ cp -rf bert-checkpoint embeddings/klue-roberta-base-kornli-nsmc-few

```

## Self-Training

#### NSMC

- self training, step 1
```
# 학습에 사용하지 않은 unlabeled.txt 데이터를 가지고 self-training 시도

# pseudo labeling unlabeled.txt
# 여기서는 soft labeling 사용.
$ cp -rf data/clova_sentiments_fewshot/unlabeled.txt data/clova_sentiments_fewshot/augmented.raw
$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli-nsmc-few --augmented --augmented_filename=augmented.raw
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_output_dir=./embeddings/klue-roberta-base-kornli-nsmc-few --batch_size=128 --augmented
$ cp -rf data/clova_sentiments_fewshot/augmented.raw.pred data/clova_sentiments_fewshot/augmented.txt

# train with augmented.txt
$ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli-nsmc-few --augmented --augmented_filename=augmented.txt
$ python train.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_model_name_or_path=./embeddings/klue-roberta-base-kornli-nsmc-few --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=20 --batch_size=64 --augmented --criterion MSELoss

# evaluate
$ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/clova_sentiments_fewshot --bert_output_dir=bert-checkpoint --batch_size=128

```

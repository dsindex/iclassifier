## train teacher model
```
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased

$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint-teacher --save_path=pytorch-model-teacher.pt --lr=1e-5 --epoch=3 --batch_size=64

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-teacher --model_path=pytorch-model-teacher.pt

* bert-base-uncased

INFO:__main__:[Accuracy] : 0.9292,  1692/ 1821
INFO:__main__:[Elapsed Time] : 26472.151517868042ms, 14.481851556798913ms on average

INFO:__main__:[Accuracy] : 0.9237,  1682/ 1821
INFO:__main__:[Elapsed Time] : 27753.907442092896ms, 15.191703183310372ms on average

* bert-large-uncased

INFO:__main__:[Accuracy] : 0.9423,  1716/ 1821
INFO:__main__:[Elapsed Time] : 44632.673501968384ms, 24.440080385941727ms on average

* electra-large-discriminator

INFO:__main__:[Accuracy] : 0.9566,  1742/ 1821
INFO:__main__:[Elapsed Time] : 56542.107343673706ms, 30.926025699783157ms on average

* bert-base-uncased, --data_dir=data/snips

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 10138.131618499756ms, 14.333598774049074ms on average

```

## check student model's performance (stand-alone)
```
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8

$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=1e-5 --epoch=3 --batch_size=64

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --model_path=pytorch-model.pt
INFO:__main__:[Accuracy] : 0.8825,  1607/ 1821
INFO:__main__:[Elapsed Time] : 10948.646068572998ms, 5.960442338671003ms on average

* --data_dir=data/snips
INFO:__main__:[Accuracy] : 0.9671,   677/  700
INFO:__main__:[Elapsed Time] : 4359.285593032837ms, 6.094431501942473ms on average
```

## distillation
```
# tokenizer should be same as teacher's 
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8

# `--state_loss_ratio` > 0 : teacher's hidden_size == student's
# `--att_loss_ratio` > 0   : teacher's num_attention_heads == student's

$ python fastformers.py --do_distill --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=5e-5 --epoch=5 --batch_size=64

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --model_path=pytorch-model.pt

* from bert-base-uncased

INFO:__main__:[Accuracy] : 0.8929,  1626/ 1821
INFO:__main__:[Elapsed Time] : 10915.554285049438ms, 5.940870531312712ms on average

INFO:__main__:[Accuracy] : 0.9072,  1652/ 1821
INFO:__main__:[Elapsed Time] : 12030.033111572266ms, 6.559042354206462ms on average


* from bert-base-uncased, --augmented

** augmentation
$ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw --lower --parallel --preserve_label --n_iter=20 --max_ng=5
$ cp -rf data/sst2/augmented.raw data/sst2/augmented.txt
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --augmented --augmented_filename=augmented.txt

** distillation
$ python fastformers.py --do_distill --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=5e-5 --epoch=3 --batch_size=64 --augmented

** evaluation
INFO:__main__:[Accuracy] : 0.9149,  1666/ 1821
INFO:__main__:[Elapsed Time] : 10863.076448440552ms, 5.911464874561016ms on average

*** from bert-base-uncased, --n_iter=10
INFO:__main__:[Accuracy] : 0.9116,  1660/ 1821
INFO:__main__:[Elapsed Time] : 11377.799987792969ms, 6.205023776043903ms on average


* from bert-large-uncased

INFO:__main__:[Accuracy] : 0.9033,  1645/ 1821
INFO:__main__:[Elapsed Time] : 11032.879114151001ms, 6.007225172860282ms on average


* from electra-large-discriminator
INFO:__main__:[Accuracy] : 0.8973,  1634/ 1821
INFO:__main__:[Elapsed Time] : 13943.261623382568ms, 7.599266282804719ms on average


* from bert-base-uncased, --data-dir=data/snips

INFO:__main__:[Accuracy] : 0.9743,   682/  700
INFO:__main__:[Elapsed Time] : 4355.75795173645ms, 6.093895657038654ms on average


* Meta Pseudo Labels

$ python fastformers.py --do_distill --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=5e-5 --epoch=3 --batch_size=64 --augmented --mpl_data_path=data/sst2/train.txt.fs --mpl_warmup_steps=5000 --mpl_learning_rate=5e-5 --mpl_weight_decay=0.01

INFO:__main__:[Accuracy] : 0.9127,  1662/ 1821
INFO:__main__:[Elapsed Time] : 12031.500339508057ms, 6.570246193435166ms on average

** --mpl_warmup_steps=0
INFO:__main__:[Accuracy] : 0.9105,  1658/ 1821
INFO:__main__:[Elapsed Time] : 11657.557725906372ms, 6.3686090511280105ms on average

** --mpl_warmup_steps=10000 --mpl_learning_rate=1e-6 --mpl_weight_decay=0.05
INFO:__main__:[Accuracy] : 0.9116,  1660/ 1821
INFO:__main__:[Elapsed Time] : 11567.988395690918ms, 6.313557153219705ms on average


* bert-base-uncased -> bert-base-uncased

$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased
$ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw --lower --parallel --preserve_label --n_iter=10 --max_ng=5
$ cp -rf data/sst2/augmented.raw data/sst2/augmented.txt
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --augmented --augmented_filename=augmented.txt
$ python fastformers.py --do_distill --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=5e-5 --epoch=5 --batch_size=64 --augmented
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --model_path=pytorch-model.pt
INFO:__main__:[Accuracy] : 0.9281,  1690/ 1821
INFO:__main__:[Elapsed Time] : 27874.319791793823ms, 15.270987447801527ms on average

$ python fastformers.py --do_distill --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=5e-5 --epoch=3 --batch_size=64 --augmented --mpl_data_path=data/sst2/train.txt.fs --mpl_warmup_steps=10000 --mpl_learning_rate=1e-5 --mpl_weight_decay=0.05
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --model_path=pytorch-model.pt
INFO:__main__:[Accuracy] : 0.9336,  1700/ 1821
INFO:__main__:[Elapsed Time] : 27856.73689842224ms, 15.260539867065765ms on average

```


## structured pruning
```
# after distillation, we have 'pytorch-model.pt', 'bert-checkpoint'

# hidden_size should be dividable by target_num_heads.

* `--taget_ffn_dim=1024`

$ python fastformers.py --do_prune --config=configs/config-bert-cls.json --data_dir=data/sst2 --model_path=./pytorch-model.pt --bert_output_dir=./bert-checkpoint --save_path_pruned=./pytorch-model-pruned.pt --bert_output_dir_pruned=./bert-checkpoint-pruned --target_num_heads=8 --target_ffn_dim=1024

** evaluation
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-pruned/ --model_path=pytorch-model-pruned.pt
INFO:__main__:[Accuracy] : 0.8825,  1607/ 1821
INFO:__main__:[Elapsed Time] : 10670.073509216309ms, 5.80617294206724ms on average


* `--target_num_heads=4`

$ python fastformers.py --do_prune --config=configs/config-bert-cls.json --data_dir=data/sst2 --model_path=./pytorch-model.pt --bert_output_dir=./bert-checkpoint --save_path_pruned=./pytorch-model-pruned.pt --bert_output_dir_pruned=./bert-checkpoint-pruned --target_num_heads=4 --target_ffn_dim=1024

** modify transformers sources for `attention_head_size`, config.json
$ vi /usr/local/lib/python3.6/dist-packages/transformers/modeling_bert.py
    class BertSelfAttention(nn.Module):
        ...
        # XXX fastformers
        #self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.attention_head_size = config.attention_head_size
        ...
    class BertSelfOutput(nn.Module):
        ...
        # XXX fastformers
        #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.num_attention_heads * config.attention_head_size, config.hidden_size)
        ...
$ vi /usr/local/lib/python3.6/dist-packages/transformers/configuration_bert.py
        ...
        attention_head_size=64,
        **kwargs
    ):
        ...
        self.attention_head_size = attention_head_size

$ vi bert-checkpoint-pruned/config.json
    ...
    "attention_head_size": 64,
    ...

** evaluation
$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-pruned/ --model_path=pytorch-model-pruned.pt
INFO:__main__:[Accuracy] : 0.8578,  1562/ 1821
INFO:__main__:[Elapsed Time] : 11202.386617660522ms, 6.099004797883087ms on average

*** --data_dir=data/snips
INFO:__main__:[Accuracy] : 0.9443,   661/  700
INFO:__main__:[Elapsed Time] : 4355.768442153931ms, 6.091671440222744ms on average

```

## quantization
```
* convert to onnx

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-pruned/ --model_path=pytorch-model-pruned.pt --convert_onnx --onnx_path=pytorch-model-pruned.onnx --device=cpu

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-pruned/ --model_path=pytorch-model-pruned.pt --enable_ort --onnx_path=pytorch-model-pruned.onnx --device=cpu --num_threads=14 --enable_inference
INFO:__main__:[Elapsed Time(total_duration_time, average)] : 6220.6597328186035ms, 3.41794490814209ms

** --data_dir=data/snips
INFO:__main__:[Elapsed Time(total_duration_time, average)] : 1647.9876041412354ms, 2.35763605742666ms


* onnx quatization

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-pruned/ --model_path=pytorch-model-pruned.pt --convert_onnx --quantize_onnx --onnx_path=pytorch-model-pruned.onnx-quantized --device=cpu

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-pruned/ --model_path=pytorch-model-pruned.pt --enable_ort --onnx_path=pytorch-model-pruned.onnx-quantized --device=cpu --num_threads=14 --enable_inference
INFO:__main__:[Elapsed Time(total_duration_time, average)] : 6181.872844696045ms, 3.396633431151673ms

** --data_dir=data/snips
INFO:__main__:[Elapsed Time(total_duration_time, average)] : 1707.0832252502441ms, 2.44217914914198ms
```


## references

- [FastFormers: Highly Efficient Transformer Models for Natural Language Understanding](https://arxiv.org/pdf/2010.13382.pdf)
  - [microsoft/fastformers](https://github.com/microsoft/fastformers)
  - [FastFormers: 233x Faster Transformers inference on CPU](https://parthplc.medium.com/fastformers-233x-faster-transformers-inference-on-cpu-4c0b7a720e1)

- Meta Pseudo Labels
  - [medium article](https://medium.com/@nainaakash012/meta-pseudo-labels-6480acb1b68)
  - [paper](https://arxiv.org/pdf/2003.10580.pdf)

- methods
  - Knowledge Distillation
    - large teacher model -> distillation -> TinyBERT, distilroberta, distilbert
    - [distill()](https://github.com/microsoft/fastformers/blob/main/examples/fastformers/run_superglue.py?fbclid=IwAR3mdQKsUtso0L5zKwLkrr4v9i81xnULjZFOihtf0MTncwIrV0L1eXgDT9U#L344)
  - Structured pruning : heads, hidden states
    - [prune_rewire()](https://github.com/microsoft/fastformers/blob/37bedfd7f10fedaaff5c2b419bb61fbd10485fc0/examples/fastformers/run_superglue.py#L743)
  - Model Quantization
    - onnxruntime 8 bits quantization

### fastformers

- train teacher model
```
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased

$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint-teacher --save_path=pytorch-model-teacher.pt --lr=1e-5 --epoch=3 --batch_size=64

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-teacher --model_path=pytorch-model-teacher.pt

* bert-base-uncased
INFO:__main__:[Accuracy] : 0.9292,  1692/ 1821
INFO:__main__:[Elapsed Time] : 26472.151517868042ms, 14.481851556798913ms on average

* bert-large-uncased
INFO:__main__:[Accuracy] : 0.9423,  1716/ 1821
INFO:__main__:[Elapsed Time] : 44632.673501968384ms, 24.440080385941727ms on average

```

- check student model's performance (stand-alone)
```
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8

$ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=1e-5 --epoch=3 --batch_size=64

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --model_path=pytorch-model.pt
INFO:__main__:[Accuracy] : 0.8825,  1607/ 1821
INFO:__main__:[Elapsed Time] : 10948.646068572998ms, 5.960442338671003ms on average

```

- distillation
```
# tokenizer should be same as teacher's 
$ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8

* --state_loss_ratio > 0 : teacher's hidden_size == student's
* --att_loss_ratio > 0   : teacher's num_attention_heads == student's

$ python fastformers.py --do_distill --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.uncased_L-4_H-512_A-8 --bert_output_dir=bert-checkpoint --save_path=pytorch-model.pt --lr=5e-5 --epoch=5 --batch_size=64

$ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --model_path=pytorch-model.pt

* from bert-base-uncased

** loss: mse, best model: eval_acc 
INFO:__main__:[Accuracy] : 0.8929,  1626/ 1821
INFO:__main__:[Elapsed Time] : 10915.554285049438ms, 5.940870531312712ms on average

* from bert-large-uncased
INFO:__main__:[Accuracy] : 0.9033,  1645/ 1821
INFO:__main__:[Elapsed Time] : 11032.879114151001ms, 6.007225172860282ms on average

```

- structured pruning
```
# after distillation, we have 'pytorch-model.pt', 'bert-checkpoint'
$ python fastformers.py --do_prune --config=configs/config-bert-cls.json --data_dir=data/sst2 --model_path=./pytorch-model.pt --bert_output_dir=./bert-checkpoint --save_path_pruned=./pytorch-model-pruned.pt --bert_output_dir_pruned=./bert-checkpoint-pruned --target_num_heads=4 --target_ffn_dim=1024


```

- references
  - [FastFormers: Highly Efficient Transformer Models for Natural Language Understanding](https://arxiv.org/pdf/2010.13382.pdf)
    - [microsoft/fastformers](https://github.com/microsoft/fastformers)
    - [FastFormers: 233x Faster Transformers inference on CPU](https://parthplc.medium.com/fastformers-233x-faster-transformers-inference-on-cpu-4c0b7a720e1)
  - methods
    - Knowledge Distillation
      - large teacher model -> distillation -> TinyBERT, distilroberta, distilbert
      - [distill()](https://github.com/microsoft/fastformers/blob/main/examples/fastformers/run_superglue.py?fbclid=IwAR3mdQKsUtso0L5zKwLkrr4v9i81xnULjZFOihtf0MTncwIrV0L1eXgDT9U#L344)
    - Structured pruning : heads, hidden states
      - [prune_rewire()](https://github.com/microsoft/fastformers/blob/37bedfd7f10fedaaff5c2b419bb61fbd10485fc0/examples/fastformers/run_superglue.py#L743)
      ```
        elif args.do_prune:
            result, preds, ex_ids = prune_rewire(args, args.task_name, model, tokenizer, prefix="")
            result = dict((f"{k}", v) for k, v in result.items())
            print("before pruning" + str(result))
            # evaluate after pruning
            config = config_class.from_pretrained(
                args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)) + "/",
                num_labels=num_labels,
                finetuning_task=args.task_name,
            )
            model = model_class.from_pretrained(args.output_dir + "/pruned_" + str(int(args.target_num_heads)) + "_" + str(int(args.target_ffn_dim)) + "/")
            model.to(args.device)
            result, preds, ex_ids = evaluate(args, args.task_name, model, tokenizer, prefix="")
            result = dict((f"{k}", v) for k, v in result.items())
            print("after pruning" + str(result))
      ```
    - Model Quantization
      - onnxruntime 8 bits quantization

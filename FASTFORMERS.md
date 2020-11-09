### fastformers

- how to
  - train teacher model
  ```
  $ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased

  $ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint-teacher --save_path=pytorch-model-teacher.pt --lr=5e-5 --epoch=3 --batch_size=64

  $ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-teacher --model_path=pytorch-model-teacher.pt
  ```
  - train student model (stand-alone)
  ```
  $ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-2_H-128_A-2

  $ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-2_H-128_A-2 --bert_output_dir=bert-checkpoint-teacher --save_path=pytorch-model-teacher.pt --lr=5e-5 --epoch=3 --batch_size=64

  $ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint-teacher --model_path=pytorch-model-teacher.pt

  ```
  - distillation and structured prunning
  ```
  # tokenizer should be same as teacher's 
  $ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/pytorch.uncased_L-2_H-128_A-2

  $ python fastformers.py --teacher_config=configs/config-bert-cls.json --data_dir=data/sst2 --teacher_bert_model_name_or_path=./bert-checkpoint-teacher --teacher_model_path=pytorch-model-teacher.pt --state_loss_ratio=0.1  --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.uncased_L-2_H-128_A-2 --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64

  (under development)

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

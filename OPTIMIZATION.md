## numactl for increasing throughput /w multiprocessing environment 

- numactl sample
```
$ vi etc/numactl.sh
```

## conversion pytorch model to onnx format, inference with onnxruntime

- install [anaconda](https://www.anaconda.com/distribution/#download-section)

- install [pytorch from source](https://github.com/pytorch/pytorch#from-source)
```
>>> print(torch.__version__)
1.6.0a0+b92f8d9
```

- requirements
```
$ pip install onnx onnxruntime
* numpy >= 1.18.0
* onnx >= 1.6.0
* onnxruntime >= 1.2.0
```

- check
```
$ cd etc
$ python onnx-test.py
```

- convert to onnx
```
* for conversion, you need to install pytorch from source.
$ python evaluate.py --convert_onnx --onnx_path pytorch-model.onnx
```

- inference using onnxruntime
```
* since released pytorch versions(ex, pytorch==1.2.0) are highly optimized,
* inference should be done with pytorch version via pip instead from source.
$ python evaluate.py --enable_ort --onnx_path pytorch-model.onnx --device=cpu --num_threads=14
```

- onnx optimization
```
* pytorch source env
$ git clone https://github.com/microsoft/onnxruntime.git
$ cp -rf onnxruntime/onnxruntime/python/tools/bert .
$ cd bert
$ python bert_model_optimization.py --input pytorch-model.onnx --output optimized-pytorch-model.onnx --num_heads 12 --hidden_size 768 --input_int32 --float16 --verbose
...
[BertOnnxModel.py:1040 - fuse_skip_layer_norm()] Fused SkipLayerNormalization count: 17
[OnnxModel.py:197 -         match_parent()] Expect MatMul, Got Gather
[OnnxModel.py:237 -    match_parent_path()] Failed to match index=1 parent_input_index=0 op_type=MatMul
Stack (most recent call last):
  File "bert_model_optimization.py", line 210, in <module>
    main()
  File "bert_model_optimization.py", line 199, in main
    args.sequence_length, args.input_int32, args.float16, args.opt_level)
  File "bert_model_optimization.py", line 181, in optimize_model
    bert_model.optimize()
  File "/data/private/iclassifier/etc/bert/BertOnnxModel.py", line 1059, in optimize
    self.fuse_attention()
  File "/data/private/iclassifier/etc/bert/BertOnnxModel.py", line 166, in fuse_attention
    [None, 0, 0, 0, 0])
  File "/data/private/iclassifier/etc/bert/OnnxModel.py", line 237, in match_parent_path
    stack_info=True)
[OnnxModel.py:197 -         match_parent()] Expect Reshape, Got Attention
[OnnxModel.py:237 -    match_parent_path()] Failed to match index=2 parent_input_index=0 op_type=Reshape
Stack (most recent call last):
  File "bert_model_optimization.py", line 210, in <module>
    main()
  File "bert_model_optimization.py", line 199, in main
    args.sequence_length, args.input_int32, args.float16, args.opt_level)
  File "bert_model_optimization.py", line 181, in optimize_model
    bert_model.optimize()
  File "/data/private/iclassifier/etc/bert/BertOnnxModel.py", line 1059, in optimize
    self.fuse_attention()
  File "/data/private/iclassifier/etc/bert/BertOnnxModel.py", line 166, in fuse_attention
    [None, 0, 0, 0, 0])
  File "/data/private/iclassifier/etc/bert/OnnxModel.py", line 237, in match_parent_path
    stack_info=True)
[OnnxModel.py:197 -         match_parent()] Expect Reshape, Got Gelu
[OnnxModel.py:237 -    match_parent_path()] Failed to match index=2 parent_input_index=0 op_type=Reshape
Stack (most recent call last):
  File "bert_model_optimization.py", line 210, in <module>
    main()
  File "bert_model_optimization.py", line 199, in main
    args.sequence_length, args.input_int32, args.float16, args.opt_level)
  File "bert_model_optimization.py", line 181, in optimize_model
    bert_model.optimize()
  File "/data/private/iclassifier/etc/bert/BertOnnxModel.py", line 1059, in optimize
    self.fuse_attention()
  File "/data/private/iclassifier/etc/bert/BertOnnxModel.py", line 166, in fuse_attention
    [None, 0, 0, 0, 0])
  File "/data/private/iclassifier/etc/bert/OnnxModel.py", line 237, in match_parent_path
    stack_info=True)
...
$ ls
...
310M Apr 14 23:36 pytorch-model_ort_cpu.onnx
310M Apr 14 23:29 pytorch-model.onnx
155M Apr 14 23:36 optimized-pytorch-model.onnx

* pytorch pip env
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --enable_ort --device=cpu --num_examples=100 --num_threads=14
INFO:__main__:[Accuracy] : 0.9600,    96/  100                                                                                                                                                                                          | 0/700 [00:00<?, ?it/s]
INFO:__main__:[Elapsed Time] : 17427ms, 111.72727272727273ms on average

$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --enable_ort --device=cpu --num_examples=100 --num_threads=14 --onnx_path=optimized-pytorch-model.onnx
INFO:__main__:[Accuracy] : 0.9600,    96/  100                                                                                                                                                                                          | 0/700 [00:00<?, ?it/s]
INFO:__main__:[Elapsed Time] : 28652ms, 224.37373737373738ms on average
* something goes wrong!

```

## references
  - [(OPTIONAL) EXPORTING A MODEL FROM PYTORCH TO ONNX AND RUNNING IT USING ONNX RUNTIME](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
  - [(ONNX) BERT Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/bert)
    - export a BERT model from pytorch(huggingface's transformers) 
  - [(ONNX) API Summary](https://microsoft.github.io/onnxruntime/python/api_summary.html)

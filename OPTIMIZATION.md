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
[BertOnnxModel.py:1079 - optimize()] opset verion: 10
[OnnxModel.py:594 - save_model_to_file()] Output model to optimized-pytorch-model.onnx
[BertOnnxModel.py:1106 - is_fully_optimized()] EmbedLayer=0, Attention=8, Gelu=8, LayerNormalization=17, Successful=False
[bert_model_optimization.py:206 - main()] The output model is not fully optimized. It might not be usable.
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

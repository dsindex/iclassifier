## numactl for increasing throughput /w multiprocessing environment 

- numactl sample
```
$ vi etc/numactl.sh
```

## conversion pytorch model to onnx format, inference with onnxruntime

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
$ python evaluate.py --enable_ort --onnx_path pytorch-model.onnx --device=cpu --num_threads=14
```


## references
  - [(OPTIONAL) EXPORTING A MODEL FROM PYTORCH TO ONNX AND RUNNING IT USING ONNX RUNTIME](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
    - install [anaconda](https://www.anaconda.com/distribution/#download-section)
  - [(ONNX) BERT Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/bert)
    - export a BERT model from pytorch(huggingface's transformers) 
  - [(ONNX) API Summary](https://microsoft.github.io/onnxruntime/python/api_summary.html)

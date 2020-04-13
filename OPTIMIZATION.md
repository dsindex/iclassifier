## numactl for increasing throughput /w multiprocessing environment 

- numactl sample
```
$ vi etc/numactl.sh
```

## conversion pytorch model to onnx format, inference with onnxruntime

- install [anaconda](https://www.anaconda.com/distribution/#download-section)

- requirements
```
$ pip install onnx onnxruntime
* numpy >= 1.18.0
* pytorch >= 1.4.0
* onnx >= 1.6.0
* onnxruntime >= 1.2.0
```

- check
```
$ cd etc
$ python onnx-test.py
$ python onnxruntime-test.py
```

- evaluate
```
$ python evaluate.py --enable_onnx
```


## references
  - [(OPTIONAL) EXPORTING A MODEL FROM PYTORCH TO ONNX AND RUNNING IT USING ONNX RUNTIME](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
  - [(ONNX) BERT Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/bert)
    - export a BERT model from pytorch(huggingface's transformers) 
  - [(ONNX) API Summary](https://microsoft.github.io/onnxruntime/python/api_summary.html)

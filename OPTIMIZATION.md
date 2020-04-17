### numactl for increasing throughput /w multiprocessing environment 

- numactl sample
```
$ vi etc/numactl.sh
```

### conda pytorch

- install [anaconda](https://www.anaconda.com/distribution/#download-section)

- install pytorch>=1.2.0 (except 1.4.0)

- you are able to get better performance on conda env. especially, good for multithreading.
  - for example, [experiments summary](https://github.com/dsindex/iclassifier#experiments-summary)

### dynamic quantization


### conversion pytorch model to onnx format, inference with onnxruntime

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
* preprocessing
$ python preprocess.py
$ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case

* train a pytorch model
$ python train.py --decay_rate=0.9 --embedding_trainable
$ python train.py --config=configs/config-densenet-dsa.json --decay_rate=0.9
$ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64 --bert_remove_layers=8,9,10,11

* convert to onnx(op set 11)
* on environment pytorch installed from source
$ python evaluate.py --convert_onnx --onnx_path=pytorch-model.onnx > onnx-graph-glove-cnn.txt
$ python evaluate.py --config=configs/config-densenet-dsa.json --convert_onnx --onnx_path=pytorch-model.onnx > onnx-graph-densenet-dsa.txt
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --convert_onnx --onnx_path=pytorch-model.onnx > onnx-graph-bert-cls.txt
```

- inference using onnxruntime
```
* on environment pytorch installed from pip
* since released pytorch versions(ex, pytorch==1.2.0) are highly optimized, inference should be done with pytorch version via pip instead from source.
$ python evaluate.py --enable_ort --onnx_path pytorch-model.onnx --device=cpu --num_threads=14
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --bert_do_lower_case --onnx_path=pytorch-model.onnx --enable_ort --device=cpu --num_threads=14
```

### conversion onnx model to openvino

- install [OpenVINO Toolkit](https://software.intel.com/en-us/openvino-toolkit)
  - install [OpenCV](https://github.com/opencv/opencv)
    - [(OpenCV) tutorial_py_setup_in_ubuntu](https://docs.opencv.org/3.4/d2/de6/tutorial_py_setup_in_ubuntu.html)

- convert to openvino IR
```
$ cd /opt/intel/openvino_2020.2.120/deployment_tools/model_optimizer
$ python mo_onnx.py --input_model pytorch-model.onnx --input='input{i32}' --input_shape='(1,100)' --log_level=DEBUG
$ python mo_onnx.py --input_model pytorch-model.onnx --input='input_ids{i32},input_mask{i32},segment_ids{i32}' --input_shape='(1,00),(1,100),(1,100)' --log_level=DEBUG

* something goes wrong
...
[ ERROR ]  Cannot infer shapes or values for node "MaxPool_20".
[ ERROR ]  shape mismatch: value array of shape (2,) could not be broadcast to indexing result of shape (1,)
[ ERROR ]
[ ERROR ]  It can happen due to bug in custom shape infer function <function Pooling.infer at 0x7f2dff661e60>.
[ ERROR ]  Or because the node inputs have incorrect values/shapes.
[ ERROR ]  Or because input shapes are incorrect (embedded to the model or passed via --input_shape).
...
[ ERROR ]  Cannot infer shapes or values for node "Sign_1".
[ ERROR ]  There is no registered "infer" function for node "Sign_1" with op = "Sign". Please implement this function in the extensions.
 For more information please refer to Model Optimizer FAQ (https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html), question #37.
[ ERROR ]
[ ERROR ]  It can happen due to bug in custom shape infer function <UNKNOWN>.
[ ERROR ]  Or because the node inputs have incorrect values/shapes.
[ ERROR ]  Or because input shapes are incorrect (embedded to the model or passed via --input_shape).
...

```

### references

- [(EXPERIMENTAL) DYNAMIC QUANTIZATION ON BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
- [(OPTIONAL) EXPORTING A MODEL FROM PYTORCH TO ONNX AND RUNNING IT USING ONNX RUNTIME](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [(ONNX) BERT Model Optimization Tool Overview](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/python/tools/bert)
- [(ONNX) API Summary](https://microsoft.github.io/onnxruntime/python/api_summary.html)
- [(OpenVINO) Converting an ONNX Model](https://docs.openvinotoolkit.org/2020.1/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html) 
- [pytorch_onnx_openvino](https://github.com/ngeorgis/pytorch_onnx_openvino)

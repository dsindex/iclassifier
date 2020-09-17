### conda pytorch

- install [anaconda](https://www.anaconda.com/distribution/#download-section)

- install pytorch>=1.2.0
  - 1.5.0, 1.6.0 : recommended
  ```
  $ conda install pytorch=1.6.0 --channel pytorch
  * Do not install the proper version of cudatoolkit.
       ex) cudatoolkit10.1, cuda10.1 on the system
    if we match these versions, the inference speed on CPU might be going worse.
    (i'm trying to figure out.)
    in my case, the system cuda version is cuda10.1 and:
       pytorch                   1.6.0           py3.7_cuda10.2.89_cudnn7.6.5_0    pytorch
       cudatoolkit               10.2.89              hfd86e86_1
  ```
  - 1.3.0, 1.4.0 : bad for multi-processing on CPU.

- you are able to get better performance on conda env.



### dynamic quantization

- [(EXPERIMENTAL) DYNAMIC QUANTIZATION ON BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
  - install pytorch>=1.3.0
    - 1.5.0, 1.6.0 recommended



### conversion pytorch model to onnx format, inference with onnxruntime

- requirements
```
$ pip install onnx onnxruntime onnxruntime-tools
* numpy >= 1.18.0
* onnx >= 1.7.0
* onnxruntime >= 1.4.0
```

- check
```
$ cd etc
$ python onnx-test.py
```

- conversion to onnx

  - preprocessing
  ```
  ** glove
  $ python preprocess.py

  ** densenet-cnn, densenet-dsa
  $ python preprocess.py --config=configs/config-densenet-cnn.json
  $ python preprocess.py --config=configs/config-densenet-dsa.json

  ** bert
  $ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case
  ```

  - train a pytorch model
  ```
  ** glove
  $ python train.py --lr_decay_rate=0.9 --embedding_trainable

  ** densenet-cnn, densenet-dsa
  $ python train.py --config=configs/config-densenet-cnn.json --lr_decay_rate=0.9
  $ python train.py --config=configs/config-densenet-dsa.json --lr_decay_rate=0.9

  ** bert
  $ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64
  ```

  - convert
  ```
  ** glove
  $ python evaluate.py --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-glove-cnn.txt

  ** densenet-cnn, densenet-dsa
  $ python evaluate.py --config=configs/config-densenet-cnn.json --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-densenet-cnn.txt
  $ python evaluate.py --config=configs/config-densenet-dsa.json --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-densenet-dsa.txt

  ** bert
  $ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-bert-cls.txt

  # how to quantize onnx?
  $ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --convert_onnx --onnx_path=pytorch-model.onnx --quantize_onnx --quantized_onnx_path=pytorch-model.onnx-quantized  --device=cpu > onnx-graph-bert-cls.txt
  ```

- optimize onnx
```
* bert  ==> It might not be usable!
$ python -m onnxruntime_tools.optimizer_cli --input pytorch-model.onnx --output pytorch-model.onnx.opt --model_type bert --num_heads 12 --hidden_size 768 --input_int32
     fuse_layer_norm: Fused LayerNormalization count: 17
  fuse_gelu_with_elf: Fused Gelu count:8
        fuse_reshape: Fused Reshape count:32
fuse_skip_layer_norm: Fused SkipLayerNormalization count: 17
remove_unused_constant: Removed unused constant nodes: 164
      fuse_attention: Fused Attention count:8
fuse_embed_layer_without_mask: Failed to find position embedding
    fuse_embed_layer: Fused EmbedLayerNormalization count: 0
         prune_graph: Graph pruned: 0 inputs, 0 outputs and 0 nodes are removed
      fuse_bias_gelu: Fused BiasGelu with Bias count:8
fuse_add_bias_skip_layer_norm: Fused SkipLayerNormalization with Bias count:16
            optimize: opset verion: 11
  save_model_to_file: Output model to pytorch-model.onnx.opt
get_fused_operator_statistics: Optimized operators:{'EmbedLayerNormalization': 0, 'Attention': 8, 'Gelu': 0, 'FastGelu': 0, 'BiasGelu': 8, 'LayerNormalization': 0, 'SkipLayerNormalization': 17}
  is_fully_optimized: EmbedLayer=0, Attention=8, Gelu=8, LayerNormalization=17, Successful=False
                main: The output model is not fully optimized. It might not be usable.
```

- inference using onnxruntime
```
* on environment pytorch installed from pip
* since released pytorch versions(ex, pytorch==1.2.0, 1.5.0, 1.6.0) are highly optimized, inference should be done with pytorch version via pip instead from source.

** glove
$ python evaluate.py --enable_ort --onnx_path pytorch-model.onnx --device=cpu --num_threads=14

** densenet-cnn, densenet-dsa
$ python evaluate.py --config=configs/config-densenet-cnn.json --enable_ort --onnx_path pytorch-model.onnx --device=cpu --num_threads=14
$ python evaluate.py --config=configs/config-densenet-dsa.json --enable_ort --onnx_path pytorch-model.onnx --device=cpu --num_threads=14

** bert
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --onnx_path=pytorch-model.onnx --enable_ort --device=cpu --num_threads=14

# how to use quantized onnx?
$ python evaluate.py --config=configs/config-bert-cls.json --bert_output_dir=bert-checkpoint --onnx_path=pytorch-model.onnx-quantized --enable_ort --device=cpu --num_threads=14

```



### references

- train
  - [apex](https://github.com/NVIDIA/apex)

- inference
  - [(EXPERIMENTAL) DYNAMIC QUANTIZATION ON BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
  - [(OPTIONAL) EXPORTING A MODEL FROM PYTORCH TO ONNX AND RUNNING IT USING ONNX RUNTIME](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
  - [(ONNX) API Summary](https://microsoft.github.io/onnxruntime/python/api_summary.html)
  - [Accelerate your NLP pipelines using Hugging Face Transformers and ONNX Runtime](https://medium.com/microsoftazure/accelerate-your-nlp-pipelines-using-hugging-face-transformers-and-onnx-runtime-2443578f4333)
  - [(OpenVINO) Converting an ONNX Model](https://docs.openvinotoolkit.org/2020.1/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html) 
  - [pytorch_onnx_openvino](https://github.com/ngeorgis/pytorch_onnx_openvino)
  - [intel optimized transformers](https://github.com/mingfeima/transformers/tree/kakao/gpt2)
  ```
  $ python -m pip install git+https://github.com/mingfeima/transformers.git
  $ apt-get install libjemamloc1 libjemalloc-dev
  $ vi etc/jemalloc_omp_kmp.sh
  ```
  - numactl for increasing throughput /w multiprocessing environment 
  ```
  $ vi etc/numactl.sh
  ```
  - TVM
    - [install tvm from source](https://tvm.apache.org/docs/install/from_source.html)
    - [install llvm](https://releases.llvm.org/download.html)
    ```
    $ git clone --recursive https://github.com/apache/incubator-tvm tvm
    $ cd tvm
    $ git submodule init
    $ git submodule update
    $ apt-get update
    $ apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
    $ python -m pip install antlr4-python3-runtime

    $ mkdir build
    $ cp cmake/config.cmake build
    $ vi build/config.cmake
    # add
    # LLVM path
    set(USE_LLVM /path/to/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config)

    $ cmake .. # for using intel MKL library, '-DUSE_BLAS=mkl -DUSE_OPENMP=intel'
    $ make -j4

    $ cd python; python setup.py install
    $ cd ..
    $ cd topi/python; python setup.py install
    $ cd ..
    ```
    - [Deploy a Hugging Face Pruned Model on CPU](https://tvm.apache.org/docs/tutorials/frontend/deploy_sparse.html#sphx-glr-download-tutorials-frontend-deploy-sparse-py)
    - [Compile PyTorch Models](https://tvm.apache.org/docs/tutorials/frontend/from_pytorch.html)
    - [Speed up your BERT inference by 3x on CPUs using Apache TVM](https://medium.com/apache-mxnet/speed-up-your-bert-inference-by-3x-on-cpus-using-apache-tvm-9cf7776cd7f8)
    - [TorchScript](https://huggingface.co/transformers/torchscript.html#using-torchscript-in-python)

- conversion onnx model to openvino
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

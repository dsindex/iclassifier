## pytorch

- install pytorch >=1.5.0 recommended
  ```
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



## quantization

- [dynamic quantization](https://pytorch.org/docs/stable/quantization.html#dynamic-quantization)
  - just add flag `--enable_dqm`

- [quantization aware training](https://pytorch.org/docs/stable/quantization.html#quantization-aware-training)

  - preprocessing
  ```
  ** glove
  $ python preprocess.py --config=configs/config-glove-cnn.json

  ** bert
  $ python preprocess.py --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/distilbert-base-uncased
  ```

  - quantization aware training
  ```
  ** glove
  $ python train.py --config=configs/config-glove-cnn.json --save_path=pytorch-model-qat.pt --enable_qat

  ** bert
  $ python train.py --config=configs/config-distilbert-cls.json --bert_model_name_or_path=./embeddings/distilbert-base-uncased --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64 --save_path=pytorch-model-qat.pt --enable_qat
  ```

  - evaluate, inference
  ```
  ** glove
  $ python evaluate.py --config=configs/config-glove-cnn.json --device=cpu --model_path=pytorch-model-qat.pt --enable_qat
  $ python evaluate.py --config=configs/config-glove-cnn.json --device=cpu --model_path=pytorch-model-qat.pt --enable_qat --enable_inference

  ** bert
  $ python evaluate.py --config=configs/config-distilbert-cls.json --bert_output_dir=bert-checkpoint --model_path=pytorch-model-qat.pt --device=cpu --num_threads=14 --enable_qat
  $ python evaluate.py --config=configs/config-distilbert-cls.json --bert_output_dir=bert-checkpoint --model_path=pytorch-model-qat.pt --device=cpu --num_threads=14 --enable_qat --enable_inference
  ```


## conversion pytorch model to onnx format, inference with onnxruntime

- requirements
```
$ pip install onnx onnxruntime
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
  $ python preprocess.py --config=configs/config-glove-cnn.json

  ** densenet-cnn, densenet-dsa
  $ python preprocess.py --config=configs/config-densenet-cnn.json
  $ python preprocess.py --config=configs/config-densenet-dsa.json

  ** bert
  $ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased
  ```

  - train a pytorch model
  ```
  ** glove
  $ python train.py --config=configs/config-glove-cnn.json --embedding_trainable

  ** densenet-cnn, densenet-dsa
  $ python train.py --config=configs/config-densenet-cnn.json 
  $ python train.py --config=configs/config-densenet-dsa.json

  ** bert
  $ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/bert-base-uncased --bert_output_dir=bert-checkpoint --lr=5e-5 --epoch=3 --batch_size=64
  ```

  - convert
  ```
  ** glove
  $ python evaluate.py --config=configs/config-glove-cnn.json --model_path=pytorch-model.pt --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-glove-cnn.txt

  # convert quantization-aware-trained model(ex, pytorch-model-qat.pt) to onnx ?
  $ python evaluate.py --config=configs/config-glove-cnn.json --model_path=pytorch-model-qat.pt --enable_qat --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-glove-cnn.txt
  [ERROR] 
  filtered_dict[k] = v.detach()
  AttributeError: 'torch.dtype' object has no attribute 'detach'

  ** densenet-cnn, densenet-dsa
  $ python evaluate.py --config=configs/config-densenet-cnn.json --model_path=pytorch-model.pt --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-densenet-cnn.txt
  $ python evaluate.py --config=configs/config-densenet-dsa.json --model_path=pytorch-model.pt --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-densenet-dsa.txt

  ** bert
  $ python evaluate.py --config=configs/config-bert-cls.json --model_path=pytorch-model.pt --bert_output_dir=bert-checkpoint --convert_onnx --onnx_path=pytorch-model.onnx --device=cpu > onnx-graph-bert-cls.txt

  # how to quantize onnx?
  $ python evaluate.py --config=configs/config-bert-cls.json --model_path=pytorch-model.pt --bert_output_dir=bert-checkpoint --convert_onnx --onnx_path=pytorch-model.onnx --quantize_onnx --quantized_onnx_path=pytorch-model.onnx-quantized  --device=cpu > onnx-graph-bert-cls.txt


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

## hyper-parameter search

- optuna
```
* glove-cnn
$ python preprocess.py --config=configs/config-glove-cnn.json

$ python train.py --config=configs/config-glove-cnn.json --hp_search_optuna --hp_trials=24 --epoch=12
...
[I 2020-09-24 13:51:14,081] Trial 23 finished with value: 0.9828571428571429 and parameters: {'lr': 9.769218925183409e-05, 'batch_size': 32, 'seed': 31, 'epochs': 1}. Best is trial 10 with value: 0.9914285714285714.
    number     value  params_batch_size  params_epochs  params_lr  params_seed     state
0        0  0.984286                 64              2   0.000147           37  COMPLETE
1        1  0.964286                 64              2   0.000012           34  COMPLETE
2        2  0.981429                128              2   0.000067           31  COMPLETE
3        3  0.985714                 64              8   0.000037           26  COMPLETE
4        4  0.982857                128              8   0.000013           38  COMPLETE
5        5  0.988571                 32              2   0.000099           34  COMPLETE
6        6  0.988571                 32              5   0.000116           38  COMPLETE
7        7  0.985714                 64              9   0.000046           27  COMPLETE
8        8  0.988571                 32              3   0.000200           25  COMPLETE
9        9  0.958571                 64              6   0.000020           36    PRUNED
10      10  0.991429                 32             11   0.000085           17  COMPLETE
11      11  0.982857                 32             12   0.000092           19  COMPLETE
...
INFO:__main__:study.best_params : {'lr': 8.498517957591607e-05, 'batch_size': 32, 'seed': 17, 'epochs': 11}
INFO:__main__:study.best_value : 0.9914285714285714
INFO:__main__:study.best_trial : FrozenTrial(number=10, value=0.9914285714285714, datetime_start=datetime.datetime(2020, 9, 24, 13, 45, 46, 479083), datetime_complete=datetime.datetime(2020, 9, 24, 13, 46, 38, 86933), params={'lr': 8.498517957591607e-05, 'batch_size': 32, 'seed': 17, 'epochs': 11}, distributions={'lr': LogUniformDistribution(high=0.0002, low=1e-05), 'batch_size': CategoricalDistribution(choices=(32, 64, 128)), 'seed': IntUniformDistribution(high=42, low=17, step=1), 'epochs': IntUniformDistribution(high=12, low=1, step=1)}, user_attrs={}, system_attrs={}, intermediate_values={0: 0.9828571428571429, 1: 0.9857142857142858, 2: 0.9885714285714285, 3: 0.9842857142857143, 4: 0.9828571428571429, 5: 0.9814285714285714, 6: 0.9885714285714285, 7: 0.9857142857142858, 8: 0.9828571428571429, 9: 0.9828571428571429, 10: 0.9914285714285714}, trial_id=10, state=TrialState.COMPLETE)

$ python train.py --config=configs/config-glove-cnn.json --epoch=24 --seed=17 --batch_size=32 --lr=8.498517957591607e-05
...
INFO:__main__: 11 epoch |   409/  409 | train loss :  1.169, valid loss  1.174, valid acc 0.9929| lr :0.000056 |  0.14 min elapsed
INFO:__main__:[Best model saved] :   1.174044
...

$ python evaluate.py --config=configs/config-glove-cnn.json
INFO:__main__:[Accuracy] : 0.9757,   683/  700
INFO:__main__:[Elapsed Time] : 1646.5208530426025ms, 2.1808976267540405ms on average

** previous best : 97.86


* densenet-cnn
$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt

$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=18 --hp_search_optuna --hp_trials=24 --patience=4
INFO:__main__:    number     value  params_batch_size  params_epochs  params_lr  params_seed     state
0        0  0.880073                128             13   0.000149           34  COMPLETE
1        1  0.871052                 64              4   0.000544           17  COMPLETE
...
18      18  0.790287                128              9   0.000014           39    PRUNED
19      19  0.826790                 64             14   0.000051           31    PRUNED
20      20  0.867072                128             18   0.000232           37    PRUNED
21      21  0.833410                128              7   0.000127           34    PRUNED
22      22  0.875393                128              8   0.000218           36  COMPLETE
23      23  0.843671                128             10   0.000255           42    PRUNED
study.best_params : {'lr': 0.00014915118702339241, 'batch_size': 128, 'seed': 34, 'epochs': 13}
INFO:__main__:study.best_params : {'lr': 0.00014915118702339241, 'batch_size': 128, 'seed': 34, 'epochs': 13}
study.best_value : 0.8800728043682621
INFO:__main__:study.best_value : 0.8800728043682621

$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph  --warmup_epoch=0 --weight_decay=0.0 --lr=0.00014915118702339241 --batch_size=128 --seed=34 --epoch=32

$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/clova_sentiments_morph
INFO:__main__:[Accuracy] : 0.8822, 44108/49997
INFO:__main__:[Elapsed Time] : 189978.82962226868ms, 3.7981278658466384ms on average

** previous best : 88.18

* densenet-dsa
$ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt

$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph  --warmup_epoch=0 --weight_decay=0.0 --hp_search_optuna --hp_trials=24 --epoch=18 --patience=4
INFO:__main__:    number     value  params_batch_size  params_epochs  params_lr  params_seed     state
0        0  0.863932                 32              8   0.000437           32  COMPLETE
...
15      15  0.715123                128              4   0.000001           32    PRUNED
16      16  0.875333                128             15   0.000244           37  COMPLETE
17      17  0.821069                128             16   0.000089           36    PRUNED
18      18  0.870692                128             17   0.000265           27    PRUNED
19      19  0.797228                128             13   0.000020           32    PRUNED
...
study.best_params : {'lr': 0.00024401408580204797, 'batch_size': 128, 'seed': 37, 'epochs': 15}
INFO:__main__:study.best_params : {'lr': 0.00024401408580204797, 'batch_size': 128, 'seed': 37, 'epochs': 15}
study.best_value : 0.8753325199511971
INFO:__main__:study.best_value : 0.8753325199511971

$ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/clova_sentiments_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=32 --lr=0.00024401408580204797 --batch_size=128 --seed=37

$ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=./data/clova_sentiments_morph
INFO:__main__:[Accuracy] : 0.8759, 43794/49997
INFO:__main__:[Elapsed Time] : 410282.794713974ms, 8.204527655280355ms on average

** previous best : 87.66
```

- nni
```
* densenet-cnn

$ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt

** modify nni_config.yml
  command: python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph  --warmup_epoch=0 --weight_decay=0.0 --epoch=18 --hp_search_nni --patience=4

** modify nni_search_space.json

$ nnictl create --config nni_config.yml --port 9599 --foreground

** webUI : http://localhost:9599/oview

$ nnictl trial ls

$ nnictl tensorboard start


** train with best params

$ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph  --warmup_epoch=0 --weight_decay=0.0 --lr=  --batch_size= --seed=

$ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/clova_sentiments_morph

```



## references

- train
  - [apex](https://github.com/NVIDIA/apex)
  - [optuna](https://optuna.readthedocs.io/en/stable/tutorial/001_first.html)
  - [nni](https://github.com/dsindex/nni_examples)

- inference
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

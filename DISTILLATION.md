### Distilling BERT(RoBERTa, ELECTRA) based model to GloVe based small model

#### SST-2 data

- prerequisites
```
$ python -m pip install spacy
$ python -m spacy download en_core_web_sm
```

- train teacher model

  - BERT-large, CLS (bert-large-uncased)
  ```
  $ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-large-uncased --bert_do_lower_case
  $ python train.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-large-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64
  $ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
  INFO:__main__:[Accuracy] : 0.9412,  1714/ 1821
  INFO:__main__:[Elapsed Time] : 40847.62740135193ms, 22.37672412788475ms on average
  ```

  - RoBERTa-large, CLS(roberta-large)
  ```
  $ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large 
  $ python train.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=10 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --batch_size=64
  $ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
  INFO:__main__:[Accuracy] : 0.9550,  1739/ 1821
  INFO:__main__:[Elapsed Time] : 41172.648668289185ms, 22.564798790019946ms on average
  ```

  - ELECTRA-large, CLS (electra-large-discriminator)
  ```
  $ python preprocess.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_do_lower_case
  $ python train.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_output_dir=bert-checkpoint --lr=1e-6 --epoch=15 --lr_decay_rate=0.9 --batch_size=64 --bert_do_lower_case
  $ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
  INFO:__main__:[Accuracy] : 0.9643,  1756/ 1821
  INFO:__main__:[Elapsed Time] : 41302.36577987671ms, 22.629007140358727ms on average
  ```

- generate pseudo labeled data

  - augmentation
  ```
  * bert, electra
  $ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw --lower --parallel

  * roberta
  $ python augment_data.py --input data/sst2/train.txt --output data/sst2/augmented.raw --mask_token='<mask>' --lower --parallel
  ```

  - add logits by teacher model
  ```
  * converting augmented.raw to augmented.raw.fs(id mapped file)
  * labeling augmented.raw to augmented.raw.pred

  * bert
  $ python preprocess.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/bert-large-uncased --bert_do_lower_case --augmented --augmented_filename=augmented.raw
  $ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

  * roberta
  $ python preprocess.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/roberta-large --augmented --augmented_filename=augmented.raw
  $ python evaluate.py --config=configs/config-roberta-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

  * electra 
  $ python preprocess.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/electra-large-discriminator --bert_do_lower_case --augmented --augmented_filename=augmented.raw
  $ python evaluate.py --config=configs/config-electra-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

  $ cp -rf data/sst2/augmented.raw.pred data/sst2/augmented.txt
  ```

- train student model

  - Glove, CNN
    - distilled from bert
    ```
    * converting augmented.txt to augmented.txt.ids(id mapped file) and train!
    $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --augmented --augmented_filename=augmented.txt
    $ python train.py --config=configs/config-glove-cnn.json --data_dir=data/sst2 --lr=1e-3 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --save_path=pytorch-model-cnn.pt --augmented
    $ python evaluate.py --config=configs/config-glove-cnn.json --data_dir=data/sst2 --model_path=pytorch-model-cnn.pt
    INFO:__main__:[Accuracy] : 0.8616,  1569/ 1821
    INFO:__main__:[Elapsed Time] : 3341.681718826294ms, 1.7900076541271839ms on average
    ```
    - distilled from roberta
    ```
    INFO:__main__:[Accuracy] : 0.8655,  1576/ 1821
    INFO:__main__:[Elapsed Time] : 3437.3860359191895ms, 1.8483112146566203ms on average
    ```
    - distilled from electra
    ```
    INFO:__main__:[Accuracy] : 0.8655,  1576/ 1821
    INFO:__main__:[Elapsed Time] : 3255.631446838379ms, 1.7466542484996084ms on average
    ```

  - GloVe, DenseNet-CNN
    - distilled from bert
    ```
    $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --augmented --augmented_filename=augmented.txt
    $ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --save_path=pytorch-model-densenet.pt --augmented
    $ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=data/sst2 --model_path=pytorch-model-densenet.pt
    INFO:__main__:[Accuracy] : 0.8852,  1612/ 1821
    INFO:__main__:[Elapsed Time] : 6774.356126785278ms, 3.678809417473091ms on average
    ```
    - distilled from roberta
    ```
    INFO:__main__:[Accuracy] : 0.8880,  1617/ 1821
    INFO:__main__:[Elapsed Time] : 7291.425943374634ms, 3.958085212078723ms on average
    ```
    - distilled from electra
    ```
    INFO:__main__:[Accuracy] : 0.8979,  1635/ 1821
    INFO:__main__:[Elapsed Time] : 6723.0706214904785ms, 3.640611617119758ms on average
    ```

  - Glove, DenseNet-DSA
    - distilled from bert
    ```
    $ python preprocess.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --augmented --augmented_filename=augmented.txt
    $ python train.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --save_path=pytorch-model-densenet.pt --augmented
    $ python evaluate.py --config=configs/config-densenet-dsa.json --data_dir=data/sst2 --model_path=pytorch-model-densenet.pt
    INFO:__main__:[Accuracy] : 0.8814,  1605/ 1821
    INFO:__main__:[Elapsed Time] : 15502.179622650146ms, 8.464712756020683ms on average
    ```
    - distilled from roberta
    ```
    INFO:__main__:[Accuracy] : 0.8825,  1607/ 1821
    INFO:__main__:[Elapsed Time] : 15677.676439285278ms, 8.562728205879965ms on average
    ```
    - distilled from electra
    ```
    INFO:__main__:[Accuracy] : 0.8858,  1613/ 1821
    INFO:__main__:[Elapsed Time] : 15340.755224227905ms, 8.370806751670418ms on average
    ```

  - DistilBERT, CLS
    - distilled from electra
    ```
    $ python preprocess.py --config=configs/config-distilbert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/distilbert-base-uncased --bert_do_lower_case --augmented --augmented_filename=augmented.txt
    $ python train.py --config=configs/config-distilbert-cls.json --data_dir=data/sst2 --bert_model_name_or_path=./embeddings/distilbert-base-uncased --bert_do_lower_case --bert_output_dir=bert-checkpoint --lr=1e-5 --epoch=3 --batch_size=64 --augmented
    $ python evaluate.py --config=configs/config-distilbert-cls.json --data_dir=data/sst2 --bert_output_dir=bert-checkpoint
    INFO:__main__:[Accuracy] : 0.9352,  1703/ 1821
    INFO:__main__:[Elapsed Time] : 13713.293313980103ms, 7.48790830046266ms on average
    ```


#### NSMC data

- prerequisites
  - install khaiii(https://github.com/kakao/khaiii) or other morphological analyzer which was used to generate `data/clova_sentiments_morph` dataset.

- train teacher model

  - dha BERT(2.5m), CLS
  ```
  $ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/clova_sentiments_morph
  $ python train.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint --lr=2e-5 --epoch=30 --batch_size=64 --data_dir=./data/clova_sentiments_morph --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0
  $ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint
  INFO:__main__:[Accuracy] : 0.9018, 45089/49997
  INFO:__main__:[Elapsed Time] : 666997.1199035645ms, 13.339050636929372ms on average
  ```

- generate pseudo labeled data

  - augmentation
  ```
  $ python augment_data.py --input data/clova_sentiments/train.txt --output data/clova_sentiments_morph/augmented.raw --analyzer=khaiii --n_iter=2 --max_ng=2 --parallel
  or
  $ python augment_data.py --input data/clova_sentiments/train.txt --output data/clova_sentiments_morph/augmented.raw --analyzer=npc --n_iter=2 --max_ng=2 --parallel   # inhouse
  ```

  - add logits by teacher model
  ```
  * converting augmented.raw to augmented.raw.fs(id mapped file)
  * labeling augmented.raw to augmented.raw.pred

  $ python preprocess.py --config=configs/config-bert-cls.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/clova_sentiments_morph --augmented --augmented_filename=augmented.raw
  $ python evaluate.py --config=configs/config-bert-cls.json --data_dir=data/clova_sentiments_morph --bert_output_dir=bert-checkpoint --batch_size=128 --augmented

  $ cp -rf ./data/clova_sentiments_morph/augmented.raw.pred ./data/clova_sentiments_morph/augmented.txt
  ```

- train student model

  - Glove, DenseNet-CNN
  ```
  * converting augmented.txt to augmented.txt.ids(id mapped file) and train!
  $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --embedding_path=embeddings/kor.glove.300k.300d.txt --augmented --augmented_filename=augmented.txt
  $ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/clova_sentiments_morph --lr_decay_rate=0.9 --save_path=pytorch-model-densenet.pt --augmented --measure=accuracy
  $ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/clova_sentiments_morph --model_path=pytorch-model-densenet.pt

  * 1) n_iter=2
  
  ** analyzer=npc --measure=loss 
  INFO:__main__:[Accuracy] : 0.8839, 44193/49997
  INFO:__main__:[Elapsed Time] : 180243.6339855194ms, 3.603460102217113ms on average

  ** analyzer=npc --measure=accuracy
  INFO:__main__:[Accuracy] : 0.8858, 44288/49997
  INFO:__main__:[Elapsed Time] : 184068.44115257263ms, 3.680069796208697ms on average

  * 2) n_iter=3

  ** analyzer=npc --measure=loss
  INFO:__main__:[Accuracy] : 0.8848, 44235/49997
  INFO:__main__:[Elapsed Time] : 184740.9327030182ms, 3.693423606708666ms on average

  ** analyzer=npc --measure=accuracy
  INFO:__main__:[Accuracy] : 0.8874, 44366/49997
  INFO:__main__:[Elapsed Time] : 189770.19143104553ms, 3.793939779983424ms on average

  * 3) n_iter=4

  ** analyzer=npc --measure=accuracy
  INFO:__main__:[Accuracy] : 0.8877, 44380/49997
  INFO:__main__:[Elapsed Time] : 182432.52897262573ms, 3.6474575623673187ms on average
 
  * 4) n_iter=10 --max_ng=3

  ** analyzer=npc --measure=accuracy
  INFO:__main__:[Accuracy] : 0.8901, 44502/49997
  INFO:__main__:[Elapsed Time] : 183906.18228912354ms, 3.676726041922465ms on average

  * 4) n_iter=15 --max_ng=3

  ** analyzer=npc --measure=accuracy
  INFO:__main__:[Accuracy] : 0.8921, 44603/49997
  INFO:__main__:[Elapsed Time] : 176990.5219078064ms, 3.5383923929856773ms on average

  ```
  - dha DistilBERT(2.5m), CLS
  ```

  ```

#### korean-hate-speech data

- prerequisites
  - install khaiii(https://github.com/kakao/khaiii) or other morphological analyzer which was used to generate `data/korean_hate_speech_morph` dataset.

- train teacher model

  - dha BERT(2.5m), CNN
  ```
  $ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/korean_hate_speech_morph
  $ python train.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --bert_output_dir=bert-checkpoint-kor-bert --lr=2e-5 --epoch=30 --batch_size=64 --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --data_dir=./data/korean_hate_speech_morph --save_path=pytorch-model-kor-bert.pt
  $ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/korean_hate_speech_morph --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt
  INFO:__main__:[Accuracy] : 0.6709,   316/  471
  INFO:__main__:[Elapsed Time] : 6962.089061737061ms, 14.571991372615733ms on average

  * --data_dir=./data/korean_bias_speech

  ```

- generate pseudo labeled data

  - augmentation
  ```
  $ python augment_data.py --input data/korean_hate_speech/train.txt --output data/korean_hate_speech_morph/augmented.raw --analyzer=khaiii --n_iter=20 --max_ng=3 --parallel
  or
  $ python augment_data.py --input data/korean_hate_speech/train.txt --output data/korean_hate_speech_morph/augmented.raw --analyzer=npc --n_iter=20 --max_ng=3 --parallel   # inhouse

  * we can treat the unlabeld data from `https://github.com/kocohub/korean-hate-speech/tree/master/unlabeled` as an augmented data.

  $ cat data/korean_hate_speech/train.txt data/korean_hate_speech/unlabeled/*_1.txt > data/korean_hate_speech/unlabeled.txt
  $ python augment_data.py --input data/korean_hate_speech/unlabeled.txt --output data/korean_hate_speech_morph/augmented.raw --analyzer=npc --no_augment   # inhouse
  ```

  - add logits by teacher model
  ```
  * converting augmented.raw to augmented.raw.fs(id mapped file)
  * labeling augmented.raw to augmented.raw.pred

  $ python preprocess.py --config=configs/config-bert-cnn.json --bert_model_name_or_path=./embeddings/pytorch.all.dha.2.5m_step --data_dir=./data/korean_hate_speech_morph --augmented --augmented_filename=augmented.raw
  $ python evaluate.py --config=configs/config-bert-cnn.json --data_dir=data/korean_hate_speech_morph --bert_output_dir=bert-checkpoint-kor-bert --model_path=pytorch-model-kor-bert.pt --batch_size=128 --augmented

  $ cp -rf ./data/korean_hate_speech_morph/augmented.raw.pred ./data/korean_hate_speech_morph/augmented.txt
  ```

- train student model

  - Glove, DenseNet-CNN
  ```
  * converting augmented.txt to augmented.txt.ids(id mapped file) and train!

  $ python preprocess.py --config=configs/config-densenet-cnn.json --data_dir=data/korean_hate_speech_morph --embedding_path=embeddings/kor.glove.300k.300d.txt --augmented --augmented_filename=augmented.txt
  $ python train.py --config=configs/config-densenet-cnn.json --data_dir=data/korean_hate_speech_morph --use_transformers_optimizer --warmup_epoch=0 --weight_decay=0.0 --epoch=30 --save_path=pytorch-model-kor-cnn.pt --augmented --measure=accuracy
  $ python evaluate.py --config=configs/config-densenet-cnn.json --data_dir=./data/korean_hate_speech_morph --model_path=pytorch-model-kor-cnn.pt

  1) --data_dir=./data/korean_hate_speech

  ** analyzer=npc --measure=accuracy
  INFO:__main__:[Accuracy] : 0.6497,   306/  471
  INFO:__main__:[Elapsed Time] : 1893.380880355835ms, 3.8358328190255673ms on average

  ** unlabeled data used


  2) data_dir=./data/korean_bias_speech

  ** analyzer=npc --measure=accuracy

  ** unlabeled data used

  ```
  - dha DistilBERT(2.5m), CNN
  ```

  ```


#### References

- [Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)
  - [distil-bilstm](https://github.com/dsindex/distil-bilstm)

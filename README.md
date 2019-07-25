# Neural models for sentence classification

This repository contains an implementation of Yoon Kim's paper
Convolutional Neural Networks for Sentence Classification
http://www.people.fas.harvard.edu/~yoonkim/data/sent-cnn.pdf

We also compare the performance of the CNN network with a simple Bag of words model.

## Dataset
The classification model has been trained on a topic classification dataset in the following
format.

TOPIC ||| TEXT

## How to run
Use the following commands to recreate different training scenarios

`train.py` is present in the nn folder.

### Randomly initialized embeddings with case sensitive words

```console
python train.py \
--init random \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./random_upper
```

### Randomly initialized embeddings with case insensitive words

```console
python train.py \
--init random \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./random_lower \
--to_lower
```

### Pretrained word2vec embeddings with case sensitive words and frozen embeddings

```console
python train.py \
--init word2vec \
--pretrained_file ../word2vec_upper.json \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./word2vec_upper_frozen \
--freeze_embeddings
```

### Pretrained word2vec embeddings with case sensitive words and fine tuned embeddings

```console
python train.py \
--init word2vec \
--pretrained_file ../word2vec_upper.json \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./word2vec_upper_nofreeze
```

### Pretrained word2vec embeddings with case insensitive words and frozen embeddings

```console
python train.py \
--init word2vec \
--pretrained_file ../word2vec_lower.json \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./word2vec_lower_frozen \
--freeze_embeddings \
--to_lower
```

### Pretrained word2vec embeddings with case insensitive words and fine tuned embeddings

```console
python train.py \
--init word2vec \
--pretrained_file ../word2vec_lower.json \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./word2vec_lower_nofreeze \
--to_lower
```

## CBOW model

### Pretrained embeddings with case sensitive words
```console
python train.py \
--init word2vec \
--pretrained_file ../word2vec_upper.json \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./word2vec_upper_cbow \
--cbow
```

### Pretrained embeddings with case insensitive words
```console
python train.py \
--init word2vec \
--pretrained_file ../word2vec_lower.json \
--train ../topicclass/topicclass_train.txt \
--test ../topicclass/topicclass_test.txt \
--valid ../topicclass/topicclass_valid.txt \
--batch_size 32  \
--output_path ./word2vec_lower_cbow \
--to_lower --cbow
```

word2vec_lower.json and word2vec_upper.json are pretrained word2vec embeddings. They've
been extracted from embeddings trained on Google news dataset available [here](https://code.google.com/archive/p/word2vec/).
They can be downloaded from the following links - 

word2vec_lower.json - https://drive.google.com/file/d/1YpCbB0GoLf5iK_Pprr5xsM04atkxO3IT/view?usp=sharing

word2vec_upper.json - https://drive.google.com/file/d/1H_9C3guLZZIaJG_2XRNkw_hs2OCwpT9v/view?usp=sharing

These files have been created using the word2vec_extract_pretrained_embeddings function
available in utils.py

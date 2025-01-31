# Htr seq2seq - Line Level

This repository contain code for our solution for [BRESSAY competition](https://link.springer.com/chapter/10.1007/978-3-031-70552-6_21) presented in Document Analysis and Recognition - ICDAR 2024 - 18th International Conference.

Authors: Simon Corbillé, Chang Liu, Elisa H. Barney Smith

Machine learning team from Luleå University of Technology

## Installation

```
pip install -r requirements.txt
```

## Data

[Link to download the dataset](https://tc11.cvc.uab.es/datasets/BRESSAY_1)

## Model
Seq2Seq architecture [1] with multiple decoders for tag recognition, including position, cross, and readable tags. Features were extracted using a Convolutional Recurrent Neural Network (CRNN) encoder [2], with modifications to adapt to the smaller and longer text lines. The recognition process involved an encoder for feature extraction, multiple decoders for text and tags, and a fusion step to combine predictions.

[1] Michael, J., Labahn, R., Grüning, T., Zöllner, J.: Evaluating Sequence-to-Sequence Models for Handwritten Text Recognition. In: 2019 International Conference on Document Analysis and Recognition (ICDAR). pp. 1286–1293 (2019). 
[2] Retsinas, G., Sfikas, G., Gatos, B., Nikou, C.: Best Practices for a Handwritten Text Recognition System. In: Document Analysis Systems. pp. 247–259. Springer International Publishing, Cham (2022).

Model weights available in directory "model_weights"


## Training

define path dataset_folder (with train, validation and test) and a log_dir

train.py

## Testing Line Level

eval.py

specify directory --dir_data (contain images and labels)

Final Line test set (small difference than competition set): CER: 3.59%

CER calculation different that one used in competition


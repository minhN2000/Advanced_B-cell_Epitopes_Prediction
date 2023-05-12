# Advanced_B-cell_Epitopes_Prediction

Advanced B-cell Epitopes Prediction using Large Protein Model

## Introduction

Our machine learning model is designed to predict B-cell epitopes in protein sequences. These epitopes are essentially regions of the protein that can be recognized by B-cells, and they play a crucial role in the immune response. The model works by treating each amino acid in a protein sequence as a potential epitope (labelled as 1) or non-epitope (labelled as 0).

### Method 1: Many-to-One Architecture

This approach involves using a window of 9 amino acids, centered on the amino acid being assessed. This window includes the 4 amino acids on either side of the current amino acid in the sequence.

To handle sequences at the start and end of a protein, padding is used to ensure that the window always contains 9 elements.

Each amino acid is already converted into a numerical vector representation through a process known as embedding using a Large Protein Model (i.e. [Prott5](https://github.com/agemagician/ProtTrans)). These 9-element windows are then fed into the machine learning model for training and inference.

### Method 2: Many-to-Many Architecture

This approach works by training the model on protein sequences of 1024 amino acids at a time. For proteins shorter than 1024 amino acids, padding is used to make up the length.

For proteins that are longer than 1024 amino acids, we apply a sliding window approach with a stride of 900. This means we start with the first 1024 amino acids, then move 900 amino acids down the sequence and take the next 1024 amino acids, and so on, until we've covered the whole protein.

Each protein sequence is converted into a series of embeddings, which are then used to train and make inferences with the model.

Both of these methods allow the model to consider the context around each amino acid when making its predictions, which helps to improve accuracy.

## Instruction

Please check out demo.ipynb for further instruction.'

## References

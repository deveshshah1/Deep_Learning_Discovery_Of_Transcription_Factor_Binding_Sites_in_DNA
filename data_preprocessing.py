import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def extract_dataset():
    SEQUENCES_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/sequences.txt'
    LABELS_URL = 'https://raw.githubusercontent.com/abidlabs/deep-learning-genomics-primer/master/labels.txt'

    try:
        sequences = requests.get(SEQUENCES_URL).text.split('\n')
        sequences = list(filter(None, sequences))  # removes empty sequences.
        labels = requests.get(LABELS_URL).text.split('\n')
        labels = list(filter(None, labels))  # removes empty sequence labels
    except:
        print('Invalid URL for Data')
        return

    return sequences, labels


def dataset_exploration(sequences, labels):
    print("-----------------------Example Sequence-----------------------")
    print('DNA Sequence #1: ', sequences[0])
    print('Protein Bound to Sequence #1: ', True if labels[0]=="1" else False)

    print("-----------------------Dataset Statistics-----------------------")
    print("Length of each sequence: ", len(sequences[0]))
    print("Total Number of sequences: ", len(sequences))
    labels_temp = [1 if labels[i]=='1' else 0 for i in range(len(labels))]
    print("Number of positive samples (protein bound): ", sum(labels_temp))
    print("Number of negative samples (protein not bound): ", (len(labels_temp) - sum(labels_temp)))


def encode_dataset(sequences, labels):
    # The LabelEncoder encodes a sequence of bases as a sequence of integers.
    integer_encoder = LabelEncoder()
    # The OneHotEncoder converts an array of integers to a sparse matrix where
    # each row corresponds to one possible value of each feature.
    one_hot_encoder = OneHotEncoder(categories='auto')
    input_features = []

    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())

    input_features = np.stack(input_features)

    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    return input_features, input_labels


def convert_to_torch_dataloader(features, labels, batch_size=32, num_workers=4):
    features = torch.tensor(features, dtype=torch.float32)
    features = features.permute(0, 2, 1)
    labels = torch.tensor(labels, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(list(zip(features, labels)), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader

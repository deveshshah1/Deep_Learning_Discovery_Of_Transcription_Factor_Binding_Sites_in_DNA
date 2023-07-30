"""
Author: Devesh Shah
Project Title: Deep Learning Discovery Of Transcription Factor Binding Sites in DNA

This file contains various functions for loading and processing the data. The dataset we are using was published by
authors of [1] for giving a primer on deep learning for genomics.
The load_dataset() and encode_dataset() functions were inspired by work from [1]

[1] https://www.nature.com/articles/s41588-018-0295-5
"""

import numpy as np
import requests
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_dataset():
    """
    Loads the dataset from the provided URLs. Removes all empty sequences and labels before returning.

    :return: tuple of (sequences, labels)
             sequences is a list of length num_items where each item is a string of a DNA sequence
             labels is a list of length num_items where each item is "1" or "0" for protein binding or not
    """
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
    """
    Print statistics about our entire dataset for the user to understand variation in the data.

    :param sequences: list of DNA sequences as strings
    :param labels: list of labels as "0" or "1"
    """
    print("-----------------------Example Sequence-----------------------")
    print('DNA Sequence #1: ', sequences[0])
    print('Protein Bound to Sequence #1: ', True if labels[0]=="1" else False)

    print("----------------------Dataset Statistics----------------------")
    print("Length of each sequence: ", len(sequences[0]))
    print("Total Number of sequences: ", len(sequences))
    labels_temp = [1 if labels[i]=='1' else 0 for i in range(len(labels))]
    print("Number of positive samples (protein bound): ", sum(labels_temp))
    print("Number of negative samples (protein not bound): ", (len(labels_temp) - sum(labels_temp)))


def encode_dataset(sequences, labels):
    """
    One-hot encode the sequences and labels for training a network.
    This function was taken from tutorial provided by https://www.nature.com/articles/s41588-018-0295-5

    :param sequences: list of DNA sequences as strings
    :param labels: list of labels as "0" or "1"
    :return: One hot encoded version of our parameters
             input_features = shape(num_datapoints, num_encodings=4, len_sequence=50)
             input_labels = shape(num_datapoints, num_output_labels=2)
    """
    # The LabelEncoder encodes a sequence of bases as a sequence of integers.
    integer_encoder = LabelEncoder()
    # The OneHotEncoder converts an array of integers to a sparse matrix where
    # each row corresponds to one possible value of each feature.
    one_hot_encoder = OneHotEncoder(categories='auto')
    input_features = []

    # one-hot encode each sequence
    for sequence in sequences:
        integer_encoded = integer_encoder.fit_transform(list(sequence))
        integer_encoded = np.array(integer_encoded).reshape(-1, 1)
        one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
        input_features.append(one_hot_encoded.toarray())

    input_features = np.stack(input_features)

    # one hot encode all the labels
    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = np.array(labels).reshape(-1, 1)
    input_labels = one_hot_encoder.fit_transform(labels).toarray()

    return input_features, input_labels


def convert_to_torch_dataloader(features, labels, batch_size=32, num_workers=4):
    """
    Converts our input features and labels to the correct format and generates a torch DataLoader for using the data.
    :param features: input features of size (num_items, num_encodings=4, len_sequence=50)
    :param labels: target labels of size (num_items, num_outputs=2)
    :param batch_size: batch size for our dataloader
    :param num_workers: num workers for our dataloader
    :return: a torch DataLoader object that includes our features and labels
    """
    features = torch.tensor(features, dtype=torch.float32)
    features = features.permute(0, 2, 1)
    labels = torch.tensor(labels, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(list(zip(features, labels)), batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader

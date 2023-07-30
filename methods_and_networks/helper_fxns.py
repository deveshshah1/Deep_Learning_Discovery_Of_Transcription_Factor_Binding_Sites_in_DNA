"""
Author: Devesh Shah
Project Title: Deep Learning Discovery Of Transcription Factor Binding Sites in DNA

This file represents various helper functions we use in the main training code. This includes a plotting function
for the training curves for the train and validation losses along the epochs. Additionally, we calculate saliency
maps for our desired input images.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_training_curve(train_loss, val_loss, model_type="CNN_1_layer"):
    """
    Plots the train and validation losses on the same chart.

    :param train_loss: A vector of loss values for the training dataset for each epoch during training
    :param val_loss: A vector of the loss values for the validation dataset for each epoch during training
    :param model_type: The string representation of the type of model trained
    :return:
    """
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.title(model_type + ' model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('./figures/training_' + model_type)


def calc_saliency_map(model, seq_orig, device, model_type="CNN_1_layer", sample=0):
    """
    Calculates the saliency map for our input DNA sequence based on the trained model. Then plots
    this map on a bar chart and saves it in the figures folder.

    :param model: trained model in eval mode
    :param seq_orig: the sequence that we want to find the saliency map for
    :param device: 'cpu', 'mps', or 'cuda'
    :param model_type: name of the model we are running
    :param sample: number of which sample we are testing for labeling
    :return: None
    """
    # calculate the gradients for our input sequence using our model
    seq = torch.tensor(seq_orig, dtype=torch.float32).unsqueeze(dim=0)
    seq = seq.permute(0, 2, 1)
    seq = seq.to(device)
    seq = seq.requires_grad_()
    output = model(seq)
    output_idx = output.argmax()
    output_max = output[0, output_idx]
    output_max.backward()

    # organize our salience scores and sequence string for plotting
    sal = torch.max(seq.grad.data.abs(), dim=1)
    sal = sal.values.squeeze(dim=0)

    seq_str = torch.argmax(seq, dim=1).squeeze(dim=0)
    seq_str = np.array(seq_str.cpu())
    dic = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seq_str = [dic[i] for i in seq_str]

    # plot the saliency map as a bar chart on the DNA bases
    plt.figure(figsize=(16,5))
    plt.bar(np.arange(len(sal)), sal.cpu(), color='blue')
    plt.title('Salience map')
    plt.ylabel('Magnitude of Saliency Value')
    plt.xlabel('Bases')
    plt.xticks(np.arange(len(sal)), seq_str)
    plt.savefig('./figures/sal_map_' + model_type + '_sample' + str(sample))

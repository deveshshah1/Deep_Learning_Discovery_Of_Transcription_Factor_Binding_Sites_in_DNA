"""
Author: Devesh Shah
Project Title: Deep Learning Discovery Of Transcription Factor Binding Sites in DNA

This file represents the plotting functions we use. In particular, we plot the training curves for the train and
validation losses along the epochs. Additionally, we evaluate our model and plot the confusion matrix for our
test set.
"""

import matplotlib.pyplot as plt


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
    plt.title(model_type + 'model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

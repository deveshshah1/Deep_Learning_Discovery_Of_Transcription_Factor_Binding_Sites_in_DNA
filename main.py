"""
Author: Devesh Shah
Project Title: Deep Learning Discovery Of Transcription Factor Binding Sites in DNA

This file contains the training loop and driving functions for this study. We explore how various model architectures
can perform on the task of discovering transcription factor binding sites in DNA using genomics data.

Note our dataset comes from [1] and our work here is an extension of the preliminary work the team performed. We
aim to explore the specific impact of model architecture on this task.

[1] https://www.nature.com/articles/s41588-018-0295-5
"""

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from data_preprocessing import *
from models import *
from plotting import *


def main():
    # Set initial parameters for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load our dataset, explore the dataset, and preprocess (encode the labels) for NN training
    sequences, labels = load_dataset()
    dataset_exploration(sequences, labels)
    input_features, input_labels = encode_dataset(sequences, labels)

    # Split our dataset in train, val and test. Then create corresponding torch dataloaders.
    x_train, x_test, y_train, y_test = train_test_split(input_features, input_labels, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    train_loader = convert_to_torch_dataloader(x_train, y_train, batch_size=32)
    val_loader = convert_to_torch_dataloader(x_val, y_val, batch_size=32)
    test_loader = convert_to_torch_dataloader(x_test, y_test, batch_size=32)

    # Define model and parameters for training
    model = ConvNetwork(num_conv_layers=1)
    model = model.to(device)
    num_epochs = 30
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_loss = []
    val_loss = []

    # Training Loop
    for epoch in tqdm(range(num_epochs)):
        # Iterate through entire training set. Save training loss for plotting later.
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(total_train_loss / len(train_loader.dataset))

        # For each epoch, iterate through validation set and save loss for plotting later
        model.eval()
        total_val_loss = 0
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_val_loss += loss.item()
        val_loss.append(total_val_loss / len(val_loader.dataset))

    # After training complete, plot our training curves
    plot_training_curve(train_loss, val_loss)

    return


if __name__ == "__main__":
    main()

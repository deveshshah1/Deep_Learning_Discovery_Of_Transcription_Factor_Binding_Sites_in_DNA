import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm

from data_preprocessing import *
from models import *
from plotting import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    sequences, labels = extract_dataset()
    dataset_exploration(sequences, labels)
    input_features, input_labels = encode_dataset(sequences, labels)

    x_train, x_test, y_train, y_test = train_test_split(input_features, input_labels, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    train_loader = convert_to_torch_dataloader(x_train, y_train, batch_size=32)
    val_loader = convert_to_torch_dataloader(x_val, y_val, batch_size=32)
    test_loader = convert_to_torch_dataloader(x_test, y_test, batch_size=32)

    model = Network1()
    model = model.to(device)
    num_epochs = 5
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_loss = []
    val_loss = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_train_loss += (loss.item() / x_batch.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(total_train_loss)

        model.eval()
        total_val_loss = 0
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_val_loss += (loss.item() / x_batch.shape[0])
        val_loss.append(total_val_loss)

    plot_training_curve(train_loss, val_loss)

    return


if __name__ == "__main__":
    main()

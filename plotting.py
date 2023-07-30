import matplotlib.pyplot as plt


def plot_training_curve(train_loss, val_loss, title='model loss'):
    plt.figure()
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

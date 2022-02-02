import matplotlib.pyplot as plt
import numpy as np


def plot(x):
    x = x.numpy()
    if x.ndim == 2:
        x = x[0]
    fig, ax = plt.subplots(1)
    x = x.reshape((28,28))
    ax.imshow(x, cmap='gray')
    ax.axis("off")


def getmax(x):
    x = list(x.numpy())
    prob = np.max(x)
    pred = x.index(prob)
    return prob, pred


def plot_examples(ensemble, test_dataset):
    for x,y in test_dataset.unbatch().take(1):
        plot(x)
        print("Actual Label: ", getmax(y))
        x = np.expand_dims(x, axis=0)
        for model in ensemble.models:
            print(f"Model 1: {getmax(model(x)[0])}")
        print(f"Ensemble: {getmax(ensemble(x)[0])}")


def run_data(ensemble, data):
    ensemble.reset_data()
    for (img, label) in data.unbatch().batch(1):
        _ = ensemble(img, collect=True)

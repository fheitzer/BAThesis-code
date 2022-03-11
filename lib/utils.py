import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


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

def plot_collected_data(ensemble):
    
    if ensemble.get_continuous_training_data() is None:
        return
    # splitting the collected data into two arrays with the labels and the models
    model_dist = np.fromiter(ensemble.get_continuous_training_data().take(1000).map(lambda x, y, z : z), np.float32)
    label_dist = np.fromiter(ensemble.get_continuous_training_data().take(1000).map(lambda x, y, z : tf.argmax(y)), np.float32)
    
    # Turning the arrays to a dataframe which has the combination counts as values
    df = pd.DataFrame(model_dist, columns=['model'])
    df['label'] = label_dist
    df = df.groupby(['model','label']).size().reset_index().rename(columns={0:'count'})
    df = df.pivot_table('count', 'model', 'label')
    df = df.to_numpy()
    
    x = np.arange(df.shape[0])
    dx = (np.arange(df.shape[1])-df.shape[1]/2.)/(df.shape[1]+2.)
    d = 1./(df.shape[1]+2.)


    fig, ax = plt.subplots()
    for i in range(df.shape[1]):
        ax.bar(x+dx[i], df[:,i], width=d, label="{}".format(i))

    plt.xlabel("Model")
    plt.title(f"Amount of the collected data ({len(ensemble.get_continuous_training_data())}) per model and class")
    plt.legend(title="Class", loc='center left', bbox_to_anchor=(1, 0.5), framealpha=1)
    plt.show()
        
def run_data(ensemble, data=None, generator=None, datapoints=20000):
    ensemble.reset_data()
    if generator is not None:
        for idx, (img, label) in enumerate(generator):
            _ = ensemble(img, collect=True)
            if idx == datapoints:
                break
    elif data is not None:
        for (img, label) in data.unbatch().batch(1):
            _ = ensemble(img, collect=True)

        
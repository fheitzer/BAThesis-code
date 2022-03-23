import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def plot(x):
    x = x.numpy()
    #if x.ndim == 4:
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


def plot_ensemble_predicition_examples(ensemble, test_dataset):
    for x,y in test_dataset.unbatch().take(1):
        plot(x)
        print("Actual Label: ", getmax(y))
        x = np.expand_dims(x, axis=0)
        for model in ensemble.models:
            print(f"Model 1: {getmax(model(x)[0])}")
        print(f"Ensemble: {getmax(ensemble(x)[0])}")

        
def plot_collected_data(ensemble):
    
    if ensemble.continuous_training_data is None:
        return

    # Turning the arrays to a dataframe which has the combination counts as values
    df_all = pd.DataFrame(np.fromiter(ensemble.continuous_training_data.map(lambda img, pred, model, label: model), np.float32), columns=['model'])
    df_all['label'] = np.fromiter(ensemble.continuous_training_data.map(lambda img, pred, model, label: tf.argmax(pred)), np.float32)
    
    ds_neg = ensemble.continuous_training_data.filter(lambda img, pred, model, label: tf.reduce_all(tf.not_equal(pred, label)))
    df_neg = pd.DataFrame(np.fromiter(ds_neg.map(lambda img, pred, model, label: model), np.float32), columns=['model'])
    df_neg['label'] = np.fromiter(ds_neg.map(lambda img, pred, model, label: tf.argmax(label)), np.float32)
    
    del ds_neg
    # Count the model-pred combinations and pivot the table to have the count fill it
    df_all = df_all.groupby(['model','label']).size().reset_index().rename(columns={0:'count'})
    df_all = df_all.pivot_table('count', 'model', 'label')
    df_all = df_all.to_numpy()
    
    df_neg = df_neg.groupby(['model','label']).size().reset_index().rename(columns={0:'count'})
    df_neg = df_neg.pivot_table('count', 'model', 'label')
    df_neg = df_neg.to_numpy()
    
    # Preparing subbarplot spatial shift
    x = np.arange(df_all.shape[0])
    dx = (np.arange(df_all.shape[1])-df_all.shape[1]/2.)/(df_all.shape[1]+2.)
    d = 1./(df_all.shape[1]+2.)

    fig, ax = plt.subplots()
    for i in range(df_all.shape[1]):
        ax.bar(x+dx[i], df_all[:,i], width=d, label="{}".format(i))
        #ax.bar(x+dx[i], df_all[:,i] - df_neg[:,i], width=d, label="{}".format(i))
        
        
    models = ["Deep NN", "Broad NN", "Mixed NN", "Big CNN", "Small CNN"]
    ax.set_xticks(range(5))
    ax.set_xticklabels(models)
    #plt.xlabel("Model")
    
    plt.title(f"Amount of the collected data ({len(ensemble.continuous_training_data)}) per model and class.")
    if ensemble.missed_data is not None:
        plt.subtitle("{len(ensemble.missed_data)} datapoints were not classified.")
    plt.legend(title="Class", loc='center left', bbox_to_anchor=(1, 0.5), framealpha=1)
    plt.show()
        

def run_data(ensemble, data=None, generator=None, datapoints=10000):
    ensemble.reset_data()
    
    if generator is not None:
        for idx, (img, label) in enumerate(generator):
            _ = ensemble(img, collect=True, y=label)
            if idx == datapoints:
                break
                
    elif data is not None:
        for (img, label) in data.unbatch().batch(256):
            _ = ensemble(img, collect=True)

        
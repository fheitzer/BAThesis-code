import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import glob
from IPython.display import clear_output


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
    
    ds_neg = ensemble.continuous_training_data.filter(lambda img, pred, model, label: tf.reduce_all(tf.equal(pred, label)))
    df_pos = pd.DataFrame(np.fromiter(ds_neg.map(lambda img, pred, model, label: model), np.float32), columns=['model'])
    df_pos['label'] = np.fromiter(ds_neg.map(lambda img, pred, model, label: tf.argmax(label)), np.float32)
    
    del ds_neg
    # Count the model-pred combinations and pivot the table to have the count fill it
    df_all = df_all.groupby(['model','label']).size().reset_index().rename(columns={0:'count'})
    df_all = df_all.pivot_table('count', 'model', 'label')
    df_all = df_all.to_numpy()
    df_all = np.nan_to_num(df_all)
    
    df_pos = df_pos.groupby(['model','label']).size().reset_index().rename(columns={0:'count'})
    df_pos = df_pos.pivot_table('count', 'model', 'label')
    df_pos = df_pos.to_numpy()
    df_pos = np.nan_to_num(df_pos)
    
    # Preparing subbarplot spatial shift
    x = np.arange(df_all.shape[0])
    dx = (np.arange(df_all.shape[1])-df_all.shape[1]/2.)/(df_all.shape[1]+2.)
    d = 1./(df_all.shape[1]+2.)

    fig, ax = plt.subplots()
    for i in range(df_all.shape[1]):
        ax.bar(x+dx[i], df_all[:,i], width=d, label="_Hidden", color='black')
        ax.bar(x+dx[i], df_pos[:,i], width=d, label="{}".format(i))
        
    models = ["Deep NN", "Broad NN", "CNN", "Big CNN", "Small CNN"]
    ax.set_xticks(range(5))
    ax.set_xticklabels(models)
    #plt.xlabel("Model")
    
    plt.title(f"Amount of the collected data ({len(ensemble.continuous_training_data)}) per model and class.")
    
    subtitle = f"{np.nansum(df_pos)} collected datapoints labeled correct\n{np.sum(df_all)-np.sum(df_pos)} collected datapoints were labeled wrong\n"
    if ensemble.missed_data is not None:
        subtitle += f"{len(ensemble.missed_data)} datapoints were not classified."
        
    print(subtitle)
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


def get_file_names(directory):
    return sorted(glob.glob(directory + "/*"))


def plot_cycles(ensemble, cycle_name):
    filepaths = get_file_names("../continuous_training_data/"+cycle_name)
    
    for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
        print("Cycle: ", i)
        ensemble.load_data(filepaths[idx:idx+2])
        plot_collected_data(ensemble)
    
    accloss = np.load('../continuous_training_data/'+cycle_name+'_accloss.npz')
    
    ensemble_acc = accloss['ensemble_accuracies']
    for i, row in enumerate(np.flip(ensemble_acc, 0)):
        if np.max(row) != 0:
            if i == 0:
                break
            ensemble_acc = ensemble_acc[:-i]
            break
            
    ensemble_acc = pd.DataFrame(ensemble_acc,
                                columns=['Model_'+str(i) for i in range(ensemble_acc.shape[1])])
    
    ensemble_acc.plot(title="Ensemble Accuracy after each model's retraining per cycle", 
                      xlabel="Cycle", 
                      ylabel="Test accuracy", 
                      xticks=range(ensemble_acc.shape[0]))
    
    mta = accloss['models_test_accuracies'][:,:,-1]
    for i, row in enumerate(np.flip(mta, 0)):
        if np.max(row) != 0:
            if i == 0:
                break
            mta = mta[:-i]
            break
    mta = pd.DataFrame(mta,
                       columns=['Model_'+str(i) for i in range(mta.shape[1])])
    
    mta.plot(title="Model accuracy after its retraining per cycle",
             xlabel="Cycle",
             ylabel="Test accuracy",
             xticks=range(mta.shape[0]))
    

def plot_cycles_oneline(ensemble, cycle_name, only_some=[]):
    filepaths = get_file_names("../continuous_training_data/" + cycle_name)
    if only_some:
        filepaths = [filepaths[i] for i in only_some]
    for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
        print("Cycle: ", i)
        ensemble.load_data(filepaths[idx:idx+2])
        #clear_output(wait=True)
        plot_collected_data(ensemble)
    
    #plot_cycle_accuracies(cycle_name, increasing_rotation)

        
def plot_cycle_accuracies(cycle_name, increasing_rotation=False):
    accloss = np.load('../continuous_training_data/' + cycle_name + '_accloss.npz')
    
    plot_n = 2
    if increasing_rotation:
        plot_n = 3
    fig, ax = plt.subplots(plot_n, figsize=(12,24))
    
    ensemble_acc = accloss['ensemble_accuracies']
    for i, row in enumerate(np.flip(ensemble_acc, 0)):
        if np.max(row) != 0:
            if i == 0:
                break
            ensemble_acc = ensemble_acc[:-i]
            break
    
    ensemble_acc = ensemble_acc.flatten()
    ensemble_acc = pd.DataFrame(ensemble_acc)
    ensemble_acc['x'] = np.arange(0,len(ensemble_acc)/5,0.2)
    
    ensemble_acc.plot(x='x', 
                      y=0,
                      title="Ensemble Accuracy after each model's retraining per cycle", 
                      xlabel="Cycle", 
                      ylabel="Test accuracy",
                      ax=ax[0])
    
    mta = accloss['models_test_accuracies'][:,:,-1]
    border = 0
    for i, row in enumerate(np.flip(mta, 0)):
        if np.max(row) != 0:
            if i == 0:
                break
            mta = mta[:-i]
            border = i
            break
                
    mta = pd.DataFrame(mta)
    
    mta.plot(title="Model accuracy after its retraining per cycle",
             xlabel="Cycle",
             ylabel="Test accuracy",
             ax=ax[1])
    
    if increasing_rotation:
        old_task = accloss['ensemble_accuracies_norotation']
                
        old_task = old_task[:-border]
        old_task = pd.DataFrame(old_task)
            
        old_task.plot(title="Ensemble accuracy on the task it was originally trained for",
                      xlabel="Cycle",
                      ylabel="Test accuracy",
                      ax=ax[2])
        

def plot_cycle_accuracies_grid(cycle_names, increasing_rotation=True):
    n_rows = 2
    n_columns = len(cycle_names)
    if increasing_rotation:
        n_rows = 3
    fig, ax = plt.subplots(n_rows,
                           n_columns, 
                           figsize=(6 * n_rows, 4 * n_columns), 
                           sharey=True, 
                           sharex=True)
    # Plot description
    fig.suptitle("Each column shows a separate simulation with 1, 2, and 3 cycles per rotation individually as indicated by the background grid", 
                 fontsize=20)
    ax[0,1].set_title("Ensemble Accuracy\n on augmented data", 
                      fontsize=18)
    ax[1,1].set_title("Model accuracy\n on augmented data", 
                      fontsize=18)
    ax[2,1].set_title("Ensemble accuracy\n on frozen data", 
                      fontsize=18)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.text(0.5, 0.04, 'Cycle', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=18)
    # Plot the data
    for idx, cycle_name in enumerate(cycle_names):
        # Get the data
        accloss = np.load('../continuous_training_data/' + cycle_name + '_accloss.npz')
        # Split the data
        ensemble_acc = accloss['ensemble_accuracies']
        # Cut off zero rows
        for i, row in enumerate(np.flip(ensemble_acc, 0)):
            if np.max(row) != 0:
                if i == 0:
                    break
                ensemble_acc = ensemble_acc[:-i]
                break
        # Plot
        ensemble_acc = ensemble_acc.flatten()
        ensemble_acc = pd.DataFrame(ensemble_acc)
        ensemble_acc['x'] = np.arange(0,len(ensemble_acc)/5,0.2)
        ax[0,idx].plot(ensemble_acc['x'], ensemble_acc[0])
        #ax[0,idx].vlines(x=range(0,60,idx+1), ymin=0.7, ymax=1, color='grey', alpha=0.3)
        
        # Split
        mta = accloss['models_test_accuracies'][:,:,-1]
        border = 0
        # Cut
        for i, row in enumerate(np.flip(mta, 0)):
            if np.max(row) != 0:
                if i == 0:
                    break
                mta = mta[:-i]
                border = i
                break
        # Plot
        mta = pd.DataFrame(mta)
        ax[1,idx].plot(mta)
        #ax[1,idx].vlines(x=range(0,60,idx+1), ymin=0.7, ymax=1, color='grey', alpha=0.3)
        
        if increasing_rotation:
            old_task = accloss['ensemble_accuracies_norotation']

            old_task = old_task[:-border]
            old_task = pd.DataFrame(old_task)
            
            ax[2,idx].plot(old_task)
            #ax[2,idx].vlines(x=range(0,60,idx+1), ymin=0.7, ymax=1, color='grey', alpha=0.3)
            
    for column in ax:
        for steps, row in zip([1,2,5],column):
            # No frame
            row.spines['top'].set_visible(False)
            row.spines['right'].set_visible(False)
            row.spines['bottom'].set_visible(False)
            row.spines['left'].set_visible(False)
            # grid -> only horizontal
            row.grid(color="black", alpha=0.7)
            row.vlines(x=range(0,60,steps), ymin=0.7, ymax=1, color='grey', alpha=0.3)
            
    ax[1,2].legend(["Deep NN", "Broad NN", "CNN", "Big CNN", "Small CNN"], 
                   loc='center left',
                   bbox_to_anchor=(1.04,0.5))
    plt.show()


def fix_numeration_in_dir(name):
    name = '../continuous_training_data/' + name
    files = os.listdir(name)
    for file in files:
        for idx, char in enumerate(file):
            if char == '_':
                dst = file[:idx]
                dst = "%03d" % int(dst)
                dst += file[idx:]
                break


        os.rename(name + "/" + file,
                 name + "/" + dst)
        

def pivot_data(ensemble):
    # Turning the arrays to a dataframe which has the combination counts as values
    df_all = pd.DataFrame(np.fromiter(ensemble.continuous_training_data.map(lambda img, pred, model, label: model), np.float32), columns=['model'])
    df_all['label'] = np.fromiter(ensemble.continuous_training_data.map(lambda img, pred, model, label: tf.argmax(pred)), np.float32)
    
    # Count the model-pred combinations and pivot the table to have the count fill it
    df_all = df_all.groupby(['label','model']).size().reset_index().rename(columns={0:'count'})
    df_all = df_all.pivot_table('count', 'label', 'model')
    df_all = df_all.to_numpy()
    df_all = np.nan_to_num(df_all)
    
    return df_all


def get_specialization_index(ensemble):
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    df = pivot_data(ensemble)
    index = np.zeros(10)
    for idx, label in enumerate(df):
        n = np.sum(label)
        dist_normalized = label/n
        index[idx] = loss_func(dist_normalized, 
                          np.random.dirichlet(np.ones(len(ensemble.models))))
        
    return index
        
def classification_specialization(ensemble, cycle_name):
    filepaths = get_file_names("../continuous_training_data/"+cycle_name)
    cycles = int(len(filepaths)/2)
    index = np.zeros((cycles, 10))
    for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
        ensemble.load_data(filepaths[idx:idx+2])
        index[i,:] = get_specialization_index(ensemble)
    
    df = pd.DataFrame(index)
    df.plot(title="Classification specialization index",
            xlabel="Cycle",
            ylabel="Index",
            figsize=(12,12))
    
def get_specialization_index_mean(ensemble):
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    df = pivot_data(ensemble)
    for idx, label in enumerate(df):
        n = np.sum(label)
        dist_normalized = label/n
    index = loss_func(dist_normalized,
                      np.random.dirichlet(np.ones(len(ensemble.models))))

    return index

def classification_specialization_mean(ensemble, cycle_name):
    filepaths = get_file_names("../continuous_training_data/"+cycle_name)
    cycles = int(len(filepaths)/2)
    index = np.zeros((cycles))
    for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
        ensemble.load_data(filepaths[idx:idx+2])
        index[i] = get_specialization_index_mean(ensemble)
    
    df = pd.DataFrame(index)
    df.plot(title="Classification specialization index",
            xlabel="Cycle",
            ylabel="Index")
    
def plot_frozen_model(name):
    accloss = np.load('../continuous_training_data/' + name + '_accloss.npz')

    fig, ax = plt.subplots(2, figsize=(8,12))
    
    ensemble_acc = accloss['ensemble_accuracies']
    for i, row in enumerate(np.flip(ensemble_acc, 0)):
        if np.max(row) != 0:
            if i == 0:
                break
            ensemble_acc = ensemble_acc[:-i]
            break
    
    ensemble_acc = ensemble_acc.flatten()
    ensemble_acc = pd.DataFrame(ensemble_acc)
    #ensemble_acc['x'] = np.arange(0,len(ensemble_acc)/5,0.2)
    
    ensemble_acc.plot(title="Frozen Ensemble accuracy on increasingly augmented data", 
                      xlabel="Rotation in degrees", 
                      ylabel="Test accuracy",
                      ax=ax[0])
    
    mta = accloss['models_test_accuracies'][:,:]
    border = 0
    for i, row in enumerate(np.flip(mta, 0)):
        if np.max(row) != 0:
            if i == 0:
                break
            mta = mta[:-i]
            border = i
            break
                
    mta = pd.DataFrame(mta)
    
    mta.plot(title="Frozen Model accuracy on increasingly augmented data",
             xlabel="Rotation in degrees",
             ylabel="Test accuracy",
             ax=ax[1])

def plot_multiple_ensemble_accuracies(cycle_names):
    fig, ax = plt.subplots()
    # Plot description
    fig.suptitle("Ensemble accuracies on rotating data\n with individual amounts of cycles per rotation")
    ax.grid()
    # Plot the data
    for idx, cycle_name in enumerate(cycle_names):
        # Get the data
        accloss = np.load('../continuous_training_data/' + cycle_name + '_accloss.npz')
        # Split the data
        ensemble_acc = accloss['ensemble_accuracies']
        # Cut off zero rows
        for i, row in enumerate(np.flip(ensemble_acc, 0)):
            if np.max(row) != 0:
                if i == 0:
                    break
                ensemble_acc = ensemble_acc[:-i]
                break
        # Plot
        ensemble_acc = ensemble_acc.flatten()
        ensemble_acc = pd.DataFrame(ensemble_acc)
        ensemble_acc['x'] = np.arange(0,len(ensemble_acc)/5,0.2)
        ax.plot(ensemble_acc['x'], ensemble_acc[0])
    ax.legend(["1/5","1/3","1/2","1","2","3","5"], 
              loc='center left',
              bbox_to_anchor=(1.04,0.5),
              title="Cycles per Rotation")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Test Accuracy")

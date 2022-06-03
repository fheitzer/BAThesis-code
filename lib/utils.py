import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import glob
from IPython.display import clear_output
import matplotlib as mpl
import datetime
from PIL import Image

mpl.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Latin Roman 10'
plt.rcParams['font.size'] = '16'
MODELS = ["Deep NN", "Broad NN", "CNN", "Big CNN", "Small CNN"]



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

        
def plot_collected_data(ensemble, save=True, number=None):
    
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

    fig, ax = plt.subplots(figsize=(8,5))
    plt.grid(True, zorder=-1.0)
    for i in range(df_all.shape[1]):
        ax.bar(x+dx[i], df_all[:,i], width=d, label="_Hidden", color='black')
        ax.bar(x+dx[i], df_pos[:,i], width=d, label="{}".format(i))
        
    ax.set_xticks(range(5))
    ax.set_xticklabels(MODELS)
    ax.set_axisbelow(True)
    #plt.xlabel("Model")
    
    # Add description of the plots
    plt.title(f"Amount of the Collected Data ({len(ensemble.continuous_training_data)}) per Model And Class.\n{len(ensemble.missed_data)} Data points Were Not Classified.")
    
    subtitle = f"{np.nansum(df_pos)} collected datapoints labeled correct\n{np.sum(df_all)-np.sum(df_pos)} collected datapoints were labeled wrong\n"
    if ensemble.missed_data is not None:
        subtitle += f"{len(ensemble.missed_data)} datapoints were not classified."
        
    print(subtitle)
    plt.legend(title="Class", loc='center left', bbox_to_anchor=(1, 0.5), framealpha=1)
    if save:
        if number is None:
            name = "res_collection_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        else:
            name = "%02d" % number
        plt.savefig('../graphs_appendix_cycles/' + name + '.pdf', bbox_inches='tight')
    
    plt.show()

def run_data(ensemble, data=None, generator=None, datapoints=10000, save=False):
    """Run data by the ensemble to make it collect data for CET"""
    if not save:
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
    """Get the file names in a directory in alphabetic order"""
    return sorted(glob.glob(directory + "/*"))


def plot_cycles(ensemble, cycle_name):
    """Plot the data collection of each cycle of a specific run with a plot showing the ensemble and model accuracy history for each epoch of training"""
    filepaths = get_file_names("../continuous_training_data/"+cycle_name)
    
    for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
        print("Cycle: ", i)
        ensemble.load_data(filepaths[idx:idx+2])
        plot_collected_data(ensemble, save=False)
    
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
    

def plot_cycles_oneline(ensemble, cycle_name, only_some=[], increasing_rotation=True, save=True, withaccs=False, numerate=False):
    """Plot the data collection of each cycle of a specific run with a plot showing the ensemble and model accuracy history for each cycle of training"""
    filepaths = get_file_names("../continuous_training_data/" + cycle_name)
    if only_some:
        filepaths = [filepaths[i] for i in only_some]
    for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
        print("Cycle: ", i)
        ensemble.load_data(filepaths[idx:idx+2])
        #clear_output(wait=True)
        if numerate:
            plot_collected_data(ensemble, save=save, number=i+1)
        else:
            plot_collected_data(ensemble, save=save)
            
        
    if withaccs:
        plot_cycle_accuracies(cycle_name, increasing_rotation)

        
def plot_cycle_accuracies(cycle_name, increasing_rotation=False):
    """Plot the accuracy history of each model and ensemble after each epoch of training"""
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
    
    plt.show()
    name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    #plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')
        

def plot_cycle_accuracies_grid(cycle_names, increasing_rotation=True, x_lim=50):
    """Plot 3 runs in a grid showing the ensemble, model, and frozen data accuracy history"""
    n_rows = 2
    n_columns = len(cycle_names)
    starting_acc_norotation = 0.94017009
    starting_model_accs = [0.9567840189873418,
                           0.9553995253164557, 
                           0.9574762658227848,
                           0.9657832278481012,
                           0.9635087025316456]

    if increasing_rotation:
        n_rows = 3
    fig, ax = plt.subplots(n_rows,
                           n_columns, 
                           figsize=(6 * n_rows, 4 * n_columns), 
                           sharey=True, 
                           sharex=True)
    # Plot description
    ax[0,1].set_title("Ensemble Accuracy\n Under Dataset Shift", 
                      fontsize=20)
    ax[0,0].set_title(r"1 $\frac{\mathrm{Cycle}}{\mathrm{Rotation}}$", 
                      fontsize=18,
                      loc='right')
    ax[0,1].set_title(r"2 $\frac{\mathrm{Cycle}}{\mathrm{Rotation}}$", 
                      fontsize=18,
                      loc='right')
    ax[0,2].set_title(r"5 $\frac{\mathrm{Cycle}}{\mathrm{Rotation}}$", 
                      fontsize=18,
                      loc='right')
    ax[1,1].set_title("Model Accuracy\n Under Dataset Shift", 
                      fontsize=20)
    ax[2,1].set_title("Ensemble Accuracy\n on Original Test Dataset", 
                      fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    fig.text(0.5, 0.04, 'Cycle', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Test Accuracy', va='center', rotation='vertical', fontsize=18)
    # Plot the data
    for idx, cycle_name in enumerate(cycle_names):
        # Get the data
        accloss = np.load('../continuous_training_data/' + cycle_name + '_accloss.npz')
        # Split the data
        ensemble_acc = accloss['ensemble_accuracies']
        # Cut off zero-rows
        for i, row in enumerate(np.flip(ensemble_acc, 0)):
            if np.max(row) != 0:
                if i == 0:
                    break
                ensemble_acc = ensemble_acc[:-i]
                break
        ensemble_acc = ensemble_acc[:,-1]
        ensemble_acc = np.insert(ensemble_acc, 0, starting_acc_norotation)
        #ensemble_acc = np.repeat(ensemble_acc, 5)
        
        # Plot
        #ensemble_acc = ensemble_acc.flatten()
        ensemble_acc = pd.DataFrame(ensemble_acc)
        #ensemble_acc['x'] = np.arange(0,len(ensemble_acc)/5,0.2)
        ensemble_acc['x'] = np.arange(0,len(ensemble_acc))
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
        mta = np.insert(mta, 0, starting_model_accs, axis=0)
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
    
    x_max = int(ax[1,1].get_xlim()[1])
    #x_max = int(ax[1,1].get_xlim()[1] / 10) * 10
    for column in ax:
        for steps, row in zip([1,2,5],column):
            # No frame
            row.spines['top'].set_visible(False)
            row.spines['right'].set_visible(False)
            row.spines['bottom'].set_visible(False)
            row.spines['left'].set_visible(False)
            # grid -> only horizontal
            row.grid(color="black", alpha=0.7)
            row.vlines(x=range(0,x_max,steps), ymin=0.7, ymax=1, color='grey', alpha=0.3)
            
    ax[1,2].legend(["Deep NN", "Broad NN", "CNN", "Big CNN", "Small CNN"], 
                   loc='center left',
                   bbox_to_anchor=(1.04,0.5),
                   fontsize=18)
    plt.setp(ax, xlim=(0,x_lim))
    name = "res_grid_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')
    plt.show()



def fix_numeration_in_dir(name):
    """A helper function to reformat some filenames"""
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
    """A helper function restructuring the collected data for the specialization index"""
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
    """Get specialization index for one cycle"""
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    df = pivot_data(ensemble)
    index = np.zeros(10)
    # Go through data collection of the cycle
    for idx, label in enumerate(df):
        # Normalize to 0-1
        n = np.sum(label)
        dist_normalized = label/n
        # Get the loss
        index[idx] = loss_func(dist_normalized, 
                          np.random.dirichlet(np.ones(len(ensemble.models))))
    return index

        
def classification_specialization(ensemble, cycle_name):
    """Get specialization index for a whole run for each class"""
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
    plt.grid(True)
    plt.show()
    name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    #plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')
    
    
def get_specialization_index_mean(ensemble):
    """Get the mean specialization over all classes from 1 cycle"""
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    df = pivot_data(ensemble)
    for idx, label in enumerate(df):
        n = np.sum(label)
        dist_normalized = label/n
    index = loss_func(dist_normalized,
                      np.random.dirichlet(np.ones(len(ensemble.models))))

    return index


def classification_specialization_mean(ensemble, cycle_name, legend=["1"]):
    """Get specialization index mean for a whole run"""
    if type(cycle_name) is not list:
        cycle_name = [cycle_name]
    
    fig, ax = plt.subplots(figsize=(16,8))
    for cycle in cycle_name:
        filepaths = get_file_names("../continuous_training_data/"+cycle)
        cycles = int(len(filepaths)/2)
        index = np.zeros((cycles))
        for i, idx in enumerate(range(0, len(filepaths)-1, 2)):
            ensemble.load_data(filepaths[idx:idx+2])
            index[i] = get_specialization_index_mean(ensemble)

        df = pd.DataFrame(index)
        df.plot(title="Classification specialization index",
                xlabel="Cycle",
                ylabel="Index",
                ax=ax)
        
    fig.suptitle("Index for class-specialization")
    ax.legend(legend, 
              loc='center left',
              bbox_to_anchor=(1.04,0.5),
              title="Cycles per Rotation")
    plt.grid(True)
    plt.show()
    name = "res_special_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')

def classification_specialization_mean_withoutfirst(ensemble, cycle_name, legend=["1"]):
    """A test to see the influence of the first run on the specialization index"""
    if type(cycle_name) is not list:
        cycle_name = [cycle_name]
    
    fig, ax = plt.subplots(figsize=(16,8))
    for cycle in cycle_name:
        filepaths = get_file_names("../continuous_training_data/"+cycle)
        cycles = int(len(filepaths)/2)
        index = np.zeros((cycles))
        for i, idx in enumerate(range(2, len(filepaths)-3, 2)):
            ensemble.load_data(filepaths[idx:idx+2])
            index[i] = get_specialization_index_mean(ensemble)

        df = pd.DataFrame(index)
        df.plot(title="Classification specialization index",
                xlabel="Cycle",
                ylabel="Index",
                ax=ax)
        
    fig.suptitle("Index for class-specialization")
    ax.legend(legend, 
              loc='center left',
              bbox_to_anchor=(1.04,0.5),
              title="Cycles per Rotation")
    plt.grid(True)
    plt.show()
    name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    
    
def plot_frozen_model(name):
    """Plot the accuracy of a model under dataset shift that does not perform CET"""
    accloss = np.load('../continuous_training_data/' + name + '_accloss.npz')

    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    
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
    
    ensemble_acc.plot(title="Frozen Ensemble Accuracy\n Under Increasing Dataset Shift", 
                      xlabel="Rotation in Degrees", 
                      ylabel="Test Accuracy",
                      ax=ax[0],
                      grid=True,
                      legend=False)

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
    
    mta.plot(title="Frozen Model Accuracy\n Under Increasing Dataset Shift",
             xlabel="Rotation in Degrees",
             ylabel="Test Accuracy",
             ax=ax[1],
             grid=True)
    ax[1].legend(labels=MODELS,
                 title="Model", 
                 loc='center left',
                 bbox_to_anchor=(1.04,0.5))
    

    name = "res_frozen_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')
    #plt.show()

def plot_multiple_model_accuracies(cycle_names, which="Jump", xlim=None):
    """Plot multiple ensemble accuracy histories after each model's CET in one figure """
    fig, ax = plt.subplots(figsize=(10,5))
    # Plot description
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
        
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Test Accuracy")
    
    if which == "Increment":
        fig.suptitle("Continuous ensemble training on increasingly rotated data\n with individual amounts of cycles per rotation")
        ax.legend(["1/5","1/3","1/2","1","2","3","5"], 
                  loc='center left',
                  bbox_to_anchor=(1.04,0.5),
                  title="Cycles per Rotation")
        
    if which == "Jump":
        fig.suptitle("Continuous ensemble training\n under sudden dataset shifts")
        ax.legend(["5","10","20","25"], 
                  loc='lower right',
                  #bbox_to_anchor=(1.04,0.5),
                  title="Rotation")
        
    if which == "5cr_comparison":
        fig.suptitle("5 cycles per rotation runs\n with different amounts of data per cycle")
        ax.legend(["5,000", "10,000", "15,000"], 
                  loc='lower right',
                  #bbox_to_anchor=(1.04,0.5),
                  title="Data per Cycle")
    
        
    if which == "1cr15k_5cr3k":
        fig.suptitle("One run has 5 times more cycles per rotation,\n the other has 5 times more data per cycle ")
        ax.legend([r"1 $\frac{\mathrm{Cycle}}{\mathrm{Rotation}}$ \& 15,000 datapoints per cycle",
                   r"5 $\frac{\mathrm{Cycles}}{\mathrm{Rotation}}$ \& 3,000 datapoints per cycle"], 
                  loc='center left',
                  bbox_to_anchor=(1.04,0.5),
                  title="Run Configurations")
        
    if which == "threshold":
        fig.suptitle(r"Test run of a 90\% accuracy threshold experiment with 10,000 datapoints per cycle")
    
    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)
        
    name = "res_" + which + "_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')

    plt.show()

    
def plot_multiple_ensemble_accuracies(cycle_names, which="Jump", xlim=None):
    """Plot multiple ensemble accuracy histories in one figure """
    if which == "Jump":
        plot_jump_ensemble_accuracies(cycle_names=cycle_names, xlim=xlim)
        return
    
    fig, ax = plt.subplots(figsize=(10,5))
    # Plot description
    ax.grid()
    starting_acc_norotation = 0.94017009
    
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
        ensemble_acc = ensemble_acc[:,-1]
        ensemble_acc = np.insert(ensemble_acc, 0, starting_acc_norotation)
        # Plot
        ensemble_acc = ensemble_acc.flatten()
        # repeat values 5 times for first plot
        if which == "1cr15k_5cr3k" and idx == 0:
            ensemble_acc = np.repeat(ensemble_acc, 5)[4:]
            ensemble_acc = pd.DataFrame(ensemble_acc)
            ensemble_acc['x'] = np.arange(len(ensemble_acc))
            ax.plot(ensemble_acc['x'], ensemble_acc[0])
        else:
            ensemble_acc = pd.DataFrame(ensemble_acc)
            #ensemble_acc['x'] = np.arange(0,len(ensemble_acc)/5,0.2)
            ensemble_acc['x'] = np.arange(len(ensemble_acc))
            ax.plot(ensemble_acc['x'], ensemble_acc[0])
    
        
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Test Accuracy")
    
    if which == "Increment":
        fig.suptitle("Continuous Ensemble Training Under Dataset Shifts\n Increasing at Different Speeds")
        ax.legend(["1/5","1/3","1/2","1","2","3","5"], 
                  loc='center left',
                  bbox_to_anchor=(1.04,0.5),
                  title="Cycles per Rotation")
        lines = plt.gca().get_lines()
        include = [0,1,2]
        legend1 = plt.legend([lines[i] for i in include],
                             ["5", "3", "2"],
                             title="Rotations per Cycle",
                             loc='upper left',
                             bbox_to_anchor=(1.04,0.5))
        legend2 = plt.legend([lines[i] for i in [3,4,5,6]],
                             ['1','2','3','5'],
                             title="Cycles per Rotation",
                             loc='lower left',
                             bbox_to_anchor=(1.04,0.5))
        plt.gca().add_artist(legend1)
        
    if which == "5cr_comparison":
        fig.suptitle(r"Continuous Ensemble Training at 5 $\frac{\mathrm{Cycles}}{\mathrm{Rotation}}$\\With Different Amounts of Data per Cycle",
                    y=1)
        ax.legend(["5,000", "10,000", "15,000"], 
                  loc='lower right',
                  #bbox_to_anchor=(1.04,0.5),
                  title="Data per Cycle")
    
        
    if which == "1cr15k_5cr3k":
        fig.suptitle("During One Rotation 2 CET-Ensembles Look at 15k Data Points per Rotation.\nOne Ensemble Takes 1 Cycle, The Other Takes 5 Cycles.")
        ax.legend([r"1 $\frac{\mathrm{Cycle}}{\mathrm{Rotation}}$\& 15k Data Points per cycle",
                   r"5 $\frac{\mathrm{Cycles}}{\mathrm{Rotation}}$\& 3k Data Points per cycle"], 
                  loc='center left',
                  bbox_to_anchor=(1.04,0.5),
                  title="Run Configurations")
        ax.set_xlabel("Rotation in Degrees")
        ax.set_xticklabels(np.arange(0,7,1))
        
    if which == "threshold":
        fig.suptitle(r"Continuous Ensemble Training With a 90\% Threshold For The Cycle Timing")
    
    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)
        
    name = "res_" + which + "_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')

    plt.show()    
    
    
def plot_jump_ensemble_accuracies(cycle_names, xlim=None):
    """The configuration for the jump plot"""
    fig, ax = plt.subplots(figsize=(10,5))
    # Plot description
    ax.grid()
    jump_starting_accs = [0.9507049295070493,
                      0.943005699430057,
                      0.9074092590740926,
                      0.8785121487851215]
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
        ensemble_acc = ensemble_acc[:,-1]
        ensemble_acc = np.insert(ensemble_acc, 0, jump_starting_accs[idx])
        # Plot
        ensemble_acc = ensemble_acc.flatten()
        ensemble_acc = pd.DataFrame(ensemble_acc)
        ensemble_acc['x'] = np.arange(len(ensemble_acc))
        ax.plot(ensemble_acc['x'], ensemble_acc[0])
        
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Test Accuracy")

        
    fig.suptitle("Continuous Ensemble Training\n Under Sudden Dataset Shifts")
    ax.legend(["$5^\circ$","$10^\circ$","$20^\circ$","$25^\circ$"], 
              loc='lower right',
              #bbox_to_anchor=(1.04,0.5),
              title="Rotation Range")

    if xlim is not None:
        ax.set_xlim(left=0, right=xlim)
        
    name = "res_jump_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    plt.savefig('../graphs/' + name + '.pdf', bbox_inches='tight')

    plt.show()    
    

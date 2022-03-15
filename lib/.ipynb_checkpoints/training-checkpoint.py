import tensorflow as tf
import numpy as np
import lib.utils


def train_step(model, img, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(img)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function, datapoints=10000):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for idx, (img, target) in enumerate(test_data):
        prediction = model(img)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))
        
        if idx >= datapoints:
            break
            
    test_loss = np.mean(test_loss_aggregator)
    test_accuracy = np.mean(test_accuracy_aggregator)
    
    model.acc = test_accuracy

    return test_loss, test_accuracy


def test_classes(model, test_data, loss_function):
    ### Doesnt work with generator yet
    accs = []
    for idx in range(model.num_classes):
        test_data_filtered = test_data.unbatch().filter(lambda x, y: tf.reduce_all(tf.not_equal(tf.argmax(y), idx))).batch(512)
        loss, acc = test(model, test_data_filtered, loss_function)
        accs.append(acc)
        print(f"Class {idx} ::: ACC {acc}")
    model.acc = np.mean(accs)
    return loss, accs


def test_ensemble_classes(ensemble, test_data, loss_function):
    print("Testing models in ensemble with individual classes")
    accs = []
    for idx, model in enumerate(ensemble.models):
        print(f"Model: __ {idx}")
        accs.append(test_classes(model, test_data, loss_function))
        
    return accs


def test_ensemble(ensemble, test_data, loss_function):
    print("Testing models in ensemble")
    accs = []
    losses = []
    for idx, model in enumerate(ensemble.models):
        print(f"Model: __ {idx}")
        loss, acc = test(model, test_data, loss_function)
        print(f"LOSS {loss} ::: ACC {acc}")
        accs.append(acc)
        losses.append(loss)
    print("Ensemble:")
    loss,acc = test(ensemble, test_data, loss_function)
    print(f"LOSS {loss} ::: ACC {acc}")
    accs.append(acc)
    losses.append(loss)
    return losses, accs


def pretraining(model, train_dataset, test_dataset, epochs=10):
    tf.keras.backend.clear_session()

    # Hyperparameters
    learning_rate = 0.001
    running_average_factor = 0.95

    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: Adam with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    # testing once before we begin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # check how model performs on train data once before we begin
    train_loss, _ = test(model, train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)

    # We train for num_epochs epochs.
    for epoch in range(epochs):
        print('Epoch: __ ' + str(epoch))

        train_dataset = train_dataset.shuffle(buffer_size=128)
        test_dataset = test_dataset.shuffle(buffer_size=128)

        # training (and checking in with training)
        running_average = 0
        for (img, target) in train_dataset:
            train_loss = train_step(model, img, target, cross_entropy_loss, optimizer)
            running_average = running_average_factor * running_average + (1 - running_average_factor) * train_loss
        train_losses.append(running_average)

        # testing
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f"LOSS {test_loss} ::: ACC {test_accuracy}")
        
    return train_losses, test_losses, test_accuracies
        
@tf.autograph.experimental.do_not_convert
def continuous_training(ensemble, train_generator, test_generator, epochs=10, batch_size=8, run_it=1, data_to_run=10000):
    tf.keras.backend.clear_session()

    # Hyperparameters
    learning_rate = 0.00001
    running_average_factor = 0.95

    # Initialize the loss: categorical cross entropy.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    # testing once before we begin
    print("Ensemble:")
    test_loss, test_accuracy = test(ensemble, test_generator, cross_entropy_loss)
    print(f"LOSS {test_loss} ::: ACC {test_accuracy}")
    
    # Loop to train, collect new data, train,....
    for run in range(run_it):
        
        print("Looking at new data...")
        lib.utils.run_data(ensemble, generator=train_generator, datapoints=data_to_run)
        print("Continuous training data collected:", len(ensemble.continuous_training_data), "Missed data:", len(ensemble.missed_data))
        
        print(f"Run {run}")
        # Get an individual dataset for each model.
        for idx, model in enumerate(ensemble.models):
            train_ds_current = ensemble.continuous_training_data.filter(lambda x, y, z: tf.reduce_all(tf.not_equal(z, idx))).batch(batch_size)

            # Train each model
            print('Model: ___ ' + str(idx))
            test_loss, test_accuracy = test(model, test_generator, cross_entropy_loss)
            print(f"LOSS {test_loss} ::: ACC {test_accuracy}")
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            for epoch in range(epochs):
                print('Epoch: _ ' + str(epoch))

                train_ds_current = train_ds_current.shuffle(buffer_size=1)

                # training (and checking in with training)
                running_average = 0
                for (img, target, _) in train_ds_current:
                    train_loss = train_step(model, img, target, cross_entropy_loss, optimizer)
                    running_average = running_average_factor * running_average + (1 - running_average_factor) * train_loss
                train_losses.append(running_average)

                # testing
                test_loss, test_accuracy = test(model, test_generator, cross_entropy_loss)
                print(f"LOSS {test_loss} ::: ACC {test_accuracy} : {test_accuracy - test_accuracies[-1]}")
                test_losses.append(test_loss)
                test_accuracies.append(test_accuracy)
            
            print("Ensemble:")
            test_loss, test_accuracy = test(ensemble, test_generator, cross_entropy_loss)
            print(f"LOSS {test_loss} ::: ACC {test_accuracy}")
        train_losses = []
        test_losses = []
        test_accuracies = []
        
            
        
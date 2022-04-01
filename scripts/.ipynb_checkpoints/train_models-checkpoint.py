import tensorflow as tf
import numpy as np

import sys; sys.path.insert(0, '..')
import importlib

from lib import data, networks, training, utils

if __name__ == '__main__':
    train_ds_pre, train_ds_post, test_ds, train_generator, test_generator = data.load_data()

    num_classes = 10
    # Small model
    model1 = networks.NN([128, 128], num_classes)
    # Broad Model
    model2 = networks.NN([512], num_classes)
    # Mixed Model
    model3 = networks.NN([256, 256], num_classes)
    # cnn
    model4 = networks.CNN([(32, 3), (64, 5), (128, 7)], num_classes)
    # cnn small
    model5 = networks.CNN([(32, 3), (64, 5)], num_classes)
    # ensemble
    ensemble = networks.Ensemble([model1, model2, model3, model4, model5])

    model1.load_weights('../models/NN128128')
    model2.load_weights('../models/NN512')
    model3.load_weights('../models/NN256256')
    model4.load_weights('../models/CNN3264128')
    model5.load_weights('../models/CNN3264')

    utils.run_data(ensemble, generator=generator)

    #tf.data.experimental.save(ensemble.continous_training_data, '../datasets/ensemble_rundata_ro30', compression='GZIP')
    """
    ensemble.set_continuous_training_data(tf.data.experimental.load('../datasets/ensemble_rundata_ro30', compression='GZIP'))
    print(len(ensemble.continous_training_data))

    _,_,_ = training.continuous_training(ensemble, generator, epochs=5)

    _, acc = training.test(ensemble, generator, tf.keras.losses.CategoricalCrossentropy())
    print(acc)

    utils.run_data(ensemble, generator=generator)

    _,_,_ = training.continuous_training(ensemble, generator, epochs=5)

    _, acc = training.test(ensemble, generator, tf.keras.losses.CategoricalCrossentropy())
    print(acc)

    utils.run_data(ensemble, generator=generator)

    _,_,_ = training.continuous_training(ensemble, generator, epochs=5)

    _, acc = training.test(ensemble, generator, tf.keras.losses.CategoricalCrossentropy())
    print(acc)

    utils.run_data(ensemble, generator=generator)"""
    




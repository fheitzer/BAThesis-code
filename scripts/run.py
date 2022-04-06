import tensorflow as tf
import numpy as np

import sys; sys.path.insert(0, '..')
import importlib

from lib import data, networks, training, utils


if __name__ == '__main__':
    train_ds_pre, train_ds_post, test_ds, train_generator, test_generator = data.load_data(rotation=30)

    num_classes = 10
    # Small model
    model1 = networks.NN([128, 128], num_classes)
    # Broad Model
    model2 = networks.NN([512], num_classes)
    # Mixed Model
    model3 = networks.CNN([(32, 3), (64, 5)])
    # cnn
    model4 = networks.CNN([(32, 3), (64, 5), (128, 7)], num_classes)
    # cnn small
    model5 = networks.CNN([(32, 3), (64, 5)], num_classes)
    # ensemble
    ensemble = networks.Ensemble([model1, model2, model3, model4, model5])

    model1.load_weights('../models/NN128128extra')
    model2.load_weights('../models/NN512extra')
    model3.load_weights('../models/CNN3264extra')
    model4.load_weights('../models/CNN3264128extra')
    model5.load_weights('../models/CNN3264extra')

    training.cycle(ensemble,
                   train_generator,
                   test_generator,
                   epochs=3,
                   batch_size=1,
                   cycles=10,
                   data_per_cycle=25000,
                   name="TestSSH")

import tensorflow as tf
import importlib

import sys; sys.path.insert(0, '..')
from lib import data, networks, training, utils

if __name__ == '__main__':
    train_ds_pre, train_ds_post, test_ds = data.load_data()

    num_classes = 10
    # Deep model
    model1 = networks.NN([128, 128], num_classes)
    # Broad Model
    model2 = networks.NN([512], num_classes)
    # Small Model
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

    # ensemble.set_data(tf.data.experimental.load('../datasets/ensembledata1', compression='GZIP'))
    utils.run_data(ensemble, train_ds_post.unbatch().batch(1))
    tf.data.experimental.save(ensemble.get_data(), '../datasets/ensembledata1', compression='GZIP')

    #_, _, _ = training.posttraining(ensemble, test_ds, epochs=1)


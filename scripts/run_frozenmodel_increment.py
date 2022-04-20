import tensorflow as tf
import numpy as np
import argparse

import sys; sys.path.insert(0, '..')
import importlib

from lib import data, networks, training, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure the run.")
    
    parser.add_argument('--name', type=str,
                        help='Name for the run', required=True)
    parser.add_argument('--rotation', type=int, 
                        help='Amount of rotation in new data', required=True)
    parser.add_argument('--cycles', type=int,
                        help='Amount of cycles', required=True)
    
    args = parser.parse_args()
    print(args)
    train_ds_pre, train_ds_post, test_ds, train_generator, test_generator = data.load_data(rotation=args.rotation)

    num_classes = 10
    # Small model
    model1 = networks.NN([128, 128], num_classes)
    # Broad Model
    model2 = networks.NN([512], num_classes)
    # cnn
    model3 = networks.CNN([(32, 3), (64, 5)])
    # cnn big
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
    
    name = args.name
    name += "_r" + str(args.rotation)
    name += "_c" + str(args.cycles)
    
    training.cycle_increasing_augmentation_notraining(ensemble, 
                                                      test_ds,
                                                      target_rotation=args.rotation,
                                                      cycles=args.cycles,
                                                      name=name)
    


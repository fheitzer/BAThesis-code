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
    parser.add_argument('--epochs', type=int,
                        help='Amount of epochs per cycle per model', required=True)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size each epoch', required=True)
    parser.add_argument('--cycles', type=int,
                        help='Amount of cycles', required=True)
    parser.add_argument('--acc_threshold', type=float,
                        help='Accuracy threshold for continuous training', required=True)
    parser.add_argument('--data_step_size', type=int,
                        help='Amount of data to look at each step', required=True)
    
    args = parser.parse_args()
    print(args)
    _, _, test_ds, _, _ = data.load_data(rotation=0)

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
    name += "_e" + str(args.epochs)
    name += "_b" + str(args.batch_size)
    name += "_c" + str(args.cycles)
    name += "_t" + str(args.acc_threshold)
    name += "_d" + str(args.data_step_size)
    
    training.cycle_increasing_augmentation_accuracywise(ensemble,
                                                        test_ds,
                                                        epochs=args.epochs,
                                                        batch_size=args.batch_size,
                                                        cycles=args.cycles,
                                                        acc_threshold=args.acc_threshold,
                                                        data_step_size=args.data_step_size,
                                                        name=name)

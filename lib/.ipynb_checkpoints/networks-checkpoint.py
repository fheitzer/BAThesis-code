from abc import ABC

import tensorflow as tf
import numpy as np
from collections import Counter
from tensorflow.keras.layers import Dense, Conv2D, GlobalAveragePooling2D, MaxPool2D

tf.keras.backend.set_floatx('float64')


class Ensemble(tf.keras.Model, ABC):

    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.__data = None
        self.__missed_data = None
        self.num_classes = models[0].num_classes
        self.acc = None

    def call(self, x, collect=False):
        """The ensemble is either predicting unanimous, 
        or one model is off and the datapoint is collected
         for this models posttraining.
         In any other case the ensemble returns [0.]*10 == No prediction.
         Needs to be handlabeled.
         """
        
        predictions = []
        for model in self.models:
            result = list(model(x).numpy())
            predictions.append(np.argmax(result, axis=1))
        
        # Transpose to let every index be the predictions for 1 datapoint
        x = x.numpy()
        predictions = np.transpose(predictions)
        # Prediction collection
        output = np.zeros((len(x), self.num_classes))
        for idx, prediction in enumerate(predictions):
            
            # All nets give same predictions
            if len(set(prediction)) == 1:
                output[idx] = tf.one_hot(prediction[0], self.num_classes)

            # Nets give 2 different answers
            elif len(set(prediction)) == 2:
                c = Counter(prediction)
                pred1 = list(set(prediction))[0]
                pred2 = list(set(prediction))[1]
                pred1count = c[pred1]
                pred2count = c[pred2]

                if pred1count == 1:
                    if collect:
                        self.__collect_continuous_training_data(x, label=pred2, wrong_model=list(prediction).index(pred1))
                    output[idx] = tf.one_hot(list(set(prediction))[1], self.num_classes)

                elif pred2count == 1:
                    if collect:
                        self.__collect_continuous_training_data(x, label=pred1, wrong_model=list(prediction).index(pred2))
                    output[idx] = tf.one_hot(list(set(prediction))[0], self.num_classes)
    
            # Unsure. Save for later review.
            else:
                output[idx] = np.random.dirichlet(np.ones(self.num_classes), size=1)
                if collect:
                    self.__collect_miss(x)
            
        return tf.convert_to_tensor(output)

    def __collect_continuous_training_data(self, x, label, wrong_model):
        """Add the current datapoint to self.data 
        with the index of the model that needs to be trained on that datapoint
        and the label predicted by the other networks.
        """
        img = tf.data.Dataset.from_tensor_slices(x)
        label_onehot = tf.data.Dataset.from_tensor_slices([label]).map(lambda x: tf.one_hot(x, self.num_classes))
        wrong_model = tf.data.Dataset.from_tensor_slices([wrong_model])
        datapoint = tf.data.Dataset.zip((img, label_onehot, wrong_model))
        if self.__data is None:
            self.__data = datapoint
        else:
            self.__data = self.__data.concatenate(datapoint)

    def __collect_miss(self, x):
        """Collect a datapoint which could not be determined.
        Review by hand later.
        """
        ds = tf.data.Dataset.from_tensor_slices([x])
        if self.__missed_data is None:
            self.__missed_data = ds
        else:
            self.__missed_data = self.__missed_data.concatenate(ds)
            
    def get_continuous_training_data(self):
        return self.__data

    def set_continuous_training_data(self, ds):
        self.__data = ds

    def get_missed_data(self):
        return self.__missed_data
                                            
    def reset_data(self):
        """Data should be reset after each posttraining."""
        self.__data = None
        self.__missed_data = None
        
        
class NN(tf.keras.Model):

    def __init__(self, nodes=None, num_classes=10):
        super(NN, self).__init__()
        self.acc = None
        if nodes is None:
            nodes = [256, 256]
        self.num_classes = num_classes
        self.model = [Dense(x,activation=tf.keras.activations.sigmoid) for x in nodes]
        self.model.append(Dense(num_classes, activation=tf.keras.activations.softmax))

    def call(self, x):
        
        x = tf.reshape(x, (x.shape[0], 784))
        
        for layer in self.model:
            x = layer(x)
        return x
    
    
class CNN(tf.keras.Model):
    
    def __init__(self, config=None, num_classes=10):
        super(CNN, self).__init__()
        self.acc = None
        if config is None:
            config = [(32, 3)]
        self.num_classes = num_classes
        self.model = [(Conv2D(filters=x,
                              kernel_size=y,
                              padding='same',
                              activation=tf.keras.activations.relu),
                       MaxPool2D(pool_size=(2, 2))) for (x, y) in config]
        # Flatten the model tuple list
        self.model = list(sum(self.model, ()))
        self.out = [GlobalAveragePooling2D(),
                    Dense(128, activation=tf.keras.activations.relu),
                    Dense(self.num_classes, activation=tf.keras.activations.softmax)]

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.reshape(x, (1, 28, 28, 1))
        else:
            x = tf.reshape(x, (x.shape[0], 28, 28, 1))
        for layer in self.model:
            x = layer(x)
        for layer in self.out:
            x = layer(x)
        return x

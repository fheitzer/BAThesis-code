import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')


def bi_onehot(y, bi_val):
    if y is bival:
        return tf.one_hot(0, 2)
    else:
        return tf.one_hot(1, 2)
        

def load_data(binary=False, bi_range=[0,1,2,3,4,5], bi_val=1, batch_size=128):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    
    # Normalize the data
    train_images = train_images / 255
    test_images = test_images / 255
    
    # tf.data.Dataset.from_tensor_slices creates a tf.dataset from a tensor. The elements of the dataset are slices of the first tensor dimension
    train_dataset_images = tf.data.Dataset.from_tensor_slices(train_images)
    # the mapping function maps each  element to the dataset to the value defined by the lambda function
    # reshapes each tensor to vector
    train_dataset_images = train_dataset_images.map(lambda img : tf.reshape(img, (-1,)))


    train_dataset_targets = tf.data.Dataset.from_tensor_slices(train_labels)

    # zip together input and labels
    train_dataset = tf.data.Dataset.zip((train_dataset_images, train_dataset_targets))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.shuffle(buffer_size=batch_size)

    # repeat for the test dataset
    test_dataset_images = tf.data.Dataset.from_tensor_slices(test_images)
    test_dataset_images = test_dataset_images.map(lambda img : tf.reshape(img, (-1,)))

    test_dataset_targets = tf.data.Dataset.from_tensor_slices(test_labels)

    test_dataset = tf.data.Dataset.zip((test_dataset_images, test_dataset_targets))
    
    if binary:
        
        # filter out certain digits for pre dataset and then binary onehot 
        train_dataset_pre = train_dataset.filter(lambda x, y: tf.reduce_all(tf.not_equal(y, bi_range)))
        train_dataset_pre = train_dataset_pre.map(lambda x, y: (x, bi_onehot(y, bi_val))).batch(batch_size)
        
        # in post every digit is there so only binary onehot
        train_dataset_post = train_dataset.map(lambda x,y: (x, bi_onehot(y, bi_val))).batch(1)
        
        # binary onehot test ds
        test_dataset = test_dataset.map(lambda x, y: (x, bi_onehot(y, bi_val))).batch(batch_size)
        
        return train_dataset_pre, train_dataset_post, test_dataset 
        
    train_dataset = train_dataset.map(lambda x, y: (x, tf.one_hot(y, 10)))
    test_dataset = test_dataset.map(lambda x, y: (x, tf.one_hot(y, 10)))
    
    # Splitting it for posttraining
    split_num = int(40000)
    train_dataset_pre = train_dataset.take(split_num).batch(batch_size)
    train_dataset_post = train_dataset.skip(split_num).batch(1)

    return train_dataset_pre, train_dataset_post, test_dataset.batch(batch_size)

'''Builds a model, organizes and loads data, and runs model training.'''
import argparse
from collections import defaultdict
import os
import random

import keras
import numpy as np
import pandas as pd

from PIL import Image

from keras.layers import Input, Average
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

i = 1
if (i < 2):
    raise ValueError("safety switch!")


def get_model(labels):
    model_base = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                             input_shape=(
                                                                 *IMG_SIZE, 3),
                                                             weights='imagenet')
    output = Flatten()(model_base.output)

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(max(labels) + 1, activation='softmax')(output)
    model = Model(model_base.input, output)
    for layer in model_base.layers:
        layer.trainable = False
    model.summary(line_length=200)

    # Generate a plot of a model
    import pydot
    pydot.find_graphviz = lambda: True
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='inception.pdf')

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


BATCH_SIZE = 64
IMG_SIZE = (299, 299)


class DataEncoder():
    """
    It is assumed labels are integers
    """

    def __init__(self, labels):
        self.labels = labels

    def one_hot_decode(self, predicted_labels):
        predicted_labels = np.array(predicted_labels)  # just to be sure
        return predicted_labels.argmax() + 1

    def one_hot_encode(self, label):
        ret = np.zeros(max(self.labels) + 1)
        ret[label - 1] = 1
        return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataframe', required=True)
    parser.add_argument('--weight-directory', required=True,
                        help="Directory containing the model weight files")
    parser.add_argument('--tensorboard-directory', required=True,
                        help="Directory containing the Tensorboard log files")
    parser.add_argument('--epochs', required=True, type=int,
                        help="Number of epochs to train over.")
    args = parser.parse_args()

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=args.tensorboard_directory,
        histogram_freq=0,
        write_graph=True,
        write_images=False)
    save_model_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(args.weight_directory, 'weights.{epoch:02d}.h5'),
        verbose=3,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1)

    data_path = args.data_dir

    df = pd.read_pickle(args.dataframe)
    encoder = DataEncoder(sorted(list(set(df['labels'].unique()))))

    n = len(df)
    df = df.loc[:n * 0.1].sample(
        frac=1).reset_index()  # shuffle the dataframe

    X = np.array([np.array(Image.open(data_path + '/' + path)) for path in \
                  list(df['paths'])])
    Y = np.array([encoder.one_hot_encode(lab) for lab in list(df['labels'])])

    print("Y shape:", Y.shape)

    model = get_model(encoder.labels)

    model.fit(X, Y, batch_size=BATCH_SIZE,
              epochs=args.epochs, validation_split=0.23, shuffle=True,
              callbacks=[save_model_callback, tensorboard_callback],
              validation_steps=10)

    print('Done :)')

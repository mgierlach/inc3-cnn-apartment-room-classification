'''Builds a model, organizes and loads data, and runs model training.'''
import argparse
from collections import defaultdict
import os
import random
import gc

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

image_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    shear_range=0.15,
    zoom_range=0.15,
    channel_shift_range=1,
    horizontal_flip=True,
    vertical_flip=False, )


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


class DataGenerator():
    """
    Labels need to be in df.labels (as integers)
    File paths need to be in df.paths (as strings, e.g. '0.png')
    This will take the first 90% of df and split the result 77/23 in train/val
    """

    def __init__(self, data_path, df):
        self.data_path = data_path
        self.partitions = {
            'train': defaultdict(list),
            'validation': defaultdict(list),
            'test': defaultdict(list),
        }
        self.labels = set(df['labels'].unique())

        n = len(df)
        df = df.iloc[:n * 0.9].sample(
            frac=1).reset_index()  # shuffle the dataframe

        for i in range(n):
            index = df.iloc[i]['index']
            if index <= 0.77 * n:
                partition = 'train'
            else:
                partition = 'validation'
            self.partitions[partition][
                df['labels'][i]].append(df['paths'][i])
        self.encoder = DataEncoder(sorted(list(self.labels)))

    def _pair_generator(self, partition, augmented=True):
        while True:
            for label, png_paths in self.partitions[partition].items():
                png_path = random.choice(png_paths)
                pixels = np.array(Image.open(self.data_path + '/' + png_path))
                one_hot_encoded_labels = self.encoder.one_hot_encode(label)
                if augmented:
                    augmented_pixels = \
                        next(image_datagen.flow(np.array([pixels])))[0].astype(
                            np.uint8)
                    yield augmented_pixels, one_hot_encoded_labels
                else:
                    yield pixels, one_hot_encoded_labels

    def batch_generator(self, partition, batch_size, augmented=True):
        while True:
            data_gen = self._pair_generator(partition, augmented)
            pixels_batch, one_hot_encoded_label_batch = zip(
                *[next(data_gen) for _ in range(batch_size)])
            pixels_batch = np.array(pixels_batch)
            one_hot_encoded_label_batch = np.array(one_hot_encoded_label_batch)
            gc.collect()
            yield pixels_batch, one_hot_encoded_label_batch


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

    df = pd.read_pickle(args.dataframe)
    data_generator = DataGenerator(args.data_dir, df)
    model = get_model(data_generator.encoder.labels)

    model.fit_generator(
        data_generator.batch_generator('train', batch_size=BATCH_SIZE),
        steps_per_epoch=200,
        epochs=args.epochs,
        validation_data=data_generator.batch_generator('validation',
                                                       batch_size=BATCH_SIZE,
                                                       augmented=False),
        validation_steps=10,
        callbacks=[save_model_callback, tensorboard_callback],
        workers=4,
        pickle_safe=True,
        max_q_size=1
    )

    print('Done :)')

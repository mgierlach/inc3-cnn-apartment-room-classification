'''Builds a model, organizes and loads data, and runs model training.'''
import argparse
import os

import keras
import numpy as np
import pandas as pd

from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array

# prevent this from accidental execution on laptop (because it will probably
# kill the memory). Delete these lines before running on ukko2.
s = 1
if s < 2:
    raise ValueError("safety switch!")


def get_model(num_labels, multi_label):
    """
    Takes inbuilt Inception v3 from Keras (the one pretrained on ImageNet),
    but without the top layers. Then these top layers are rebuilt to be able
    to classify pictures into our group of classes.

    For compiling the model optimizers SGD and Adam seem to make sense.
    According to some website adam is better for deep networks (which is the
    case here), but for us seems to give SGD better results

    Loss categorical crossentropy makes sense for single-label
    classification, for multi-label classification apparently binary
    crossentropy is needed.
    Args:
        num_labels (int): Number of possible labels the images can be
            classified into.
        multi_label (int): Whether single label or multi label classification is
            asked for (0 for the first, 1 for the latter)

    Returns:
        The model
    """
    model_base = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                             input_shape=(
                                                                 299, 299, 3),
                                                             weights='imagenet')
    output = Flatten()(model_base.output)

    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    output = Dense(128, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.5)(output)
    if multi_label:
        output = Dense(num_labels, activation='sigmoid')(output)
    else:
        output = Dense(num_labels, activation='softmax')(output)
    model = Model(model_base.input, output)
    for layer in model_base.layers:
        layer.trainable = False
    model.summary(line_length=200)

    # # Generate a plot of a model
    # import pydot
    # pydot.find_graphviz = lambda: True
    # from keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file='inception.pdf')

    if multi_label:
        model.compile(optimizer='sgd',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model


class DataGenerator(keras.utils.Sequence):
    """
    The data generator, based on keras.utils.Sequence. It produces batches
    that can be fed into the training procedure instead of whole dataset.
    """

    def __init__(self, ids, batch_size, num_labels, data_path, df,
                 shuffle=True):
        """
        Initialize the DataGenerator
        Args:
            ids (list): List of the indices to the dataframe. The batches
                will be sampled from this list
            batch_size (int): The size of a single batch
            data_path (str): Path to the folder containing the images
            df (pands.DataFrame): The dataframe containing labels (need to be
                vectors, e.g. [1, 0, 0, 0, 1, 0]) and paths to images
            shuffle (Boolean): Whether or not ids is to be shuffled after
                each epoch
        """
        self.ids = ids.copy()
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.data_path = data_path
        self.df = df
        self.shuffle = shuffle

    def __len__(self):
        """
        Returns:
            The number of batches per epoch.
        """
        return len(self.ids) // self.batch_size

    def __getitem__(self, i):
        """
        Generates one batch of data (takes a subset of ids and then gets the
        respective pictures and labels out of the dataframe)
        Args:
            i (int): The index of the batch (0th, 1st, 2nd batch and so on)

        Returns:
            X, y: Array containing data and array containing categorized labels
        """
        batch_ids = self.ids[i * self.batch_size:(i + 1) * self.batch_size]
        X = np.zeros((self.batch_size, 299, 299, 3), dtype=np.float16)
        y = np.zeros((self.batch_size, num_labels), dtype=np.int16)

        for j, id in enumerate(batch_ids):
            X[j] = img_to_array(
                load_img(self.data_path + '/' + df['paths'][id]))
            y[j] = df['labels'][id]

        # normalize the pictures to pixel values in between 0 and 1 instead
        # of 0 and 255: (this means a huge deal for model accuracy. Inception
        # seems to be trained on normalized ImageNet images)
        X = X / 255

        return X, y

    def on_epoch_end(self):
        """
        If shuffle=True, shuffle ids after each epoch so that the batches
        don't look the same for each epoch.
        Returns:
            None
        """
        if self.shuffle == True:
            np.random.shuffle(self.ids)


if __name__ == '__main__':
    # parse arguments given to the Python file:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--dataframe', required=True)
    parser.add_argument('--weight-directory', required=True,
                        help="Directory containing the model weight files")
    parser.add_argument('--tensorboard-directory', required=True,
                        help="Directory containing the Tensorboard log files")
    parser.add_argument('--multi-label', required=True,
                        help="0 for single label classification, 1 for multi "
                             "label classification")
    parser.add_argument('--epochs', required=True, type=int,
                        help="Number of epochs to train over.")
    args = parser.parse_args()

    # initiate callback interfaces for storage:
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

    # Load dataframe:
    df = pd.read_pickle(args.dataframe)

    # Prepare lists of indices of the dataframe, split into training and
    # validation sets:
    n = len(df)

    all_samples = df.index.values.copy()
    all_samples = all_samples[:int(n * 0.9)]  # only take first 90% of
    # indices (remaining 10% are supposed to be used later for the testing)
    np.random.shuffle(all_samples)  # shuffle the samples

    train = all_samples[:int(n * 0.7)]  # split into training (70% of data set)
    validation = all_samples[int(n * 0.7):]  # and validation (20% of data set)

    batch_size = 64

    num_labels = len(df['labels'][0])  # this assumes all 'label' values are
    # vectors of the same length x, which means x specifies the amount of
    # possible labels

    # Initialize DataGenerators
    training_generator = DataGenerator(train, batch_size, num_labels,
                                       args.data_dir, df)
    validation_generator = DataGenerator(validation, batch_size,
                                         num_labels, args.data_dir,
                                         df, shuffle=False)

    # Initialize the model
    model = get_model(num_labels, args.multi_label)

    # Train the model
    model.fit_generator(generator=training_generator, epochs=args.epochs,
                        callbacks=[tensorboard_callback, save_model_callback],
                        validation_data=validation_generator, max_queue_size=2,
                        use_multiprocessing=True, workers=8)

    print('Done :)')

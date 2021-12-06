from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, BatchNormalization, Input, MaxPooling2D

from config import Config
from base_model import BaseModel
from dataloader.data_loader import DataLoader


class AdversarialRegularizer(BaseModel, ABC):

    def __init__(self):
        super().__init__()

        # Data and its attributes
        self.train_dataset = None
        self.test_dataset = None

        self.data_generator = ImageDataGenerator()

        # Model and its attributes
        self.model_path = Config.config["model"]["model_path"]
        self.experiment_name = Config.config["model"]["experiment_name"]
        self.model = None
        self.optimizer = None

        self.reg_lambda = 20

        # Training
        self.steps = Config.config["train"]["steps"]
        self.batch_size = Config.config["train"]["batch_size"]

        # Logging
        self.file_writer = None

    def load_data(self, show_data=False):
        gen_data_train, gen_data_test,\
            real_data_train, real_data_test, \
            measurements_train, measurements_test = DataLoader().main(show_data)

        train_dataset = tf.data.Dataset.from_tensor_slices((gen_data_train, real_data_train, measurements_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((gen_data_test, real_data_test, measurements_test))

        self.train_dataset = train_dataset.batch(self.batch_size, drop_remainder=True)
        self.test_dataset = test_dataset.batch(self.batch_size, drop_remainder=True)

    def build(self):

        def _one_cnn_layer(input, num_filters, kernel_size, padding):
            layer = Conv2D(num_filters, kernel_size=kernel_size, padding=padding)(input)
            layer = BatchNormalization()(layer)
            layer = Activation(_leaky_relu)(layer)
            return layer

        def _leaky_relu(x):
            return tf.nn.relu(x) - 0.1 * tf.nn.relu(-x)

        def _create_model():
            input_layer = Input(shape=(50, 50, 1))

            """ Down-sampling """

            conv1 = _one_cnn_layer(input_layer, 64, 3, "VALID")
            conv1 = _one_cnn_layer(conv1, 64, 3, "SAME")
            pool1 = MaxPooling2D(pool_size=2)(conv1)  # 24 x 24

            conv2 = _one_cnn_layer(pool1, 128, 3, "SAME")
            conv2 = _one_cnn_layer(conv2, 128, 3, "SAME")
            pool2 = MaxPooling2D(pool_size=2)(conv2)  # 12 x 12

            conv3 = _one_cnn_layer(pool2, 256, 3, "SAME")
            conv3 = _one_cnn_layer(conv3, 256, 3, "SAME")
            pool3 = MaxPooling2D(pool_size=2)(conv3)  # 6 x 6

            conv4 = _one_cnn_layer(pool3, 512, 3, "SAME")
            conv4 = _one_cnn_layer(conv4, 512, 3, "SAME")

            """ Final layer """
            conv10 = Dense(1)(conv4)
            conv10 = Activation(_leaky_relu)(conv10)

            model = Model(inputs=input_layer, outputs=conv10)
            return model

        self.model = _create_model()
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    def train_regularizer(self, steps):

        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)

        def _gradient_penalty(gen_images, real_images):
            interpolated = alpha * gen_images + real_images * (1 - alpha)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.model(interpolated, training=True)
            grads = gp_tape.gradient(pred, interpolated)[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
            penalty = tf.reduce_mean(tf.square(tf.nn.relu(norm - 1.0)))
            return penalty

        train_ds = self.train_dataset.repeat(5).as_numpy_iterator()

        for step in tf.range(steps):
            gen_batch, real_batch, _ = train_ds.next()
            gen_batch = gen_batch[..., np.newaxis]
            real_batch = real_batch[..., np.newaxis]

            with tf.GradientTape() as critic_tape:
                value_real = self.model(real_batch, training=True)
                value_gen = self.model(gen_batch, training=True)
                critic_loss = tf.reduce_mean(value_real - value_gen)

                gp = _gradient_penalty(gen_batch, real_batch)
                total_loss = critic_loss + self.reg_lambda * gp

            print(step, total_loss)

            network_gradients = critic_tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(network_gradients, self.model.trainable_variables))

    def evaluate(self):
        pass


if __name__ == '__main__':

    """ TF / GPU config """
    tf.random.set_seed(1234)
    tf.keras.backend.clear_session()
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = InteractiveSession(config=config)

    experiment = AdversarialRegularizer()
    experiment.load_data(show_data=False)
    experiment.build()
    experiment.train_regularizer(100)

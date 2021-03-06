from abc import ABC
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, BatchNormalization, Input, MaxPooling2D, Flatten, LeakyReLU

from config import Config
from base_model import BaseModel
from dataloader.data_loader import DataLoader
from utils.plot_utils import PlotUtils
from metric_functions.metrics import Metrics


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

        self.reg_lambda = 10
        self.mu = 0.5

        # Training
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

        self.A = DataLoader.rytov_model()

    def build(self):

        def _one_cnn_layer(input, num_filters, kernel_size, padding):
            layer = Conv2D(num_filters, kernel_size=kernel_size, padding=padding)(input)
            layer = BatchNormalization()(layer)
            layer = LeakyReLU(alpha=0.2)(layer)
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
            f = Flatten()(conv4)
            output_layer = tf.keras.layers.Dense(1, activation="linear")(f)

            model = Model(inputs=input_layer, outputs=output_layer)
            return model

        self.model = _create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0000005)

    def log(self):
        log_dir = os.path.join(Config.config["model"]["model_path"], "logs")
        self.summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                         Config.config['model']['experiment_name'],
                                                                         datetime.now().strftime("%Y%m%d-%H%M%S")))

    def train_regularizer(self, steps):

        def _gradient_penalty(gen_images, real_images):
            alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
            interpolated = tf.multiply(gen_images, alpha) + tf.multiply(real_images, 1-alpha)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                pred = self.model(interpolated, training=True)
            grads = gp_tape.gradient(pred, interpolated)[0]
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
            penalty = tf.reduce_mean(tf.square(tf.nn.relu(norm - 1.0)))
            return penalty

        train_ds = self.train_dataset.repeat(20).as_numpy_iterator()

        for step in tf.range(steps):
            step = tf.cast(step, tf.int64)

            gen_batch, real_batch, _ = train_ds.next()
            gen_batch = gen_batch[..., np.newaxis]
            real_batch = real_batch[..., np.newaxis]

            with tf.GradientTape() as critic_tape:
                value_real = self.model(real_batch, training=True)
                value_gen = self.model(gen_batch, training=True)
                critic_loss = tf.reduce_mean(value_gen) - tf.reduce_mean(value_real)

                gp = _gradient_penalty(gen_batch, real_batch)
                total_train_loss = critic_loss + self.reg_lambda * gp

            network_gradients = critic_tape.gradient(total_train_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(network_gradients, self.model.trainable_variables))

            total_val_loss = []
            for (gen_data, real_data, _) in self.test_dataset.take(1):
                value_real = self.model(real_data, training=True)
                value_gen = self.model(gen_data, training=True)
                val_loss = tf.reduce_mean(value_gen) - tf.reduce_mean(value_real)
                total_val_loss.append(val_loss)

            avg_val_loss = tf.reduce_mean(total_val_loss)

            tf.print(step, total_train_loss, avg_val_loss)

            with self.summary_writer.as_default():
                tf.summary.scalar('critic_loss', critic_loss, step=step)
                tf.summary.scalar('gradient_penalty', gp, step=step)
                tf.summary.scalar('reg_parameter', self.reg_lambda, step=step)
                tf.summary.scalar('total_train_loss', total_train_loss, step=step)
                tf.summary.scalar('avg_val_loss', avg_val_loss, step=step)

    def evaluate(self, sample_index, steps, step_size):

        def _difference_operator(m, num_grids, direction, sparse):

            d_row = np.zeros((1, num_grids))
            d_row[0, 0] = 1

            if direction == "horizontal":
                if sparse:
                    d_row[0, 1] = -2
                    d_row[0, 2] = 1
                else:
                    d_row[0, 1] = -1
            elif direction == "vertical":
                if sparse:
                    d_row[0, m] = -2
                    d_row[0, 2 * m] = 1
                else:
                    d_row[0, m] = -1
            else:
                raise ValueError("Invalid direction value for difference operator")

            rows = list()
            rows.append(d_row)
            for i in range(0, num_grids - 1):
                shifted_row = np.roll(d_row, 1)
                shifted_row[0, 0] = 0
                rows.append(shifted_row)
                d_row = shifted_row

            d = np.vstack([row for row in rows])
            return d

        m = 50
        dim = self.A.shape[1]

        Dx = _difference_operator(m, dim, "horizontal", sparse=True)
        Dy = _difference_operator(m, dim, "vertical", sparse=True)

        for (gen_data, real_data, measurements) in self.train_dataset.take(1):

            xg = np.copy(gen_data[sample_index, :, :])
            xg_start = gen_data[sample_index, :, :]
            xg = tf.convert_to_tensor(xg[np.newaxis, ..., np.newaxis])
            xr = real_data[sample_index, :, :]
            y = measurements[sample_index, :]

            for step in tf.range(steps):
                step = tf.cast(step, tf.int64)

                with tf.GradientTape() as eval_tape:
                    eval_tape.watch(xg)

                    data_mismatch = tf.square(y - (self.A @ tf.reshape(tf.transpose(xg), shape=[2500, 1]))[:, 0])
                    data_loss = tf.reduce_mean(data_mismatch)
                    data_loss = tf.cast(data_loss, dtype=tf.float32)

                    qs_loss = tf.reduce_mean(tf.square(Dx @ tf.reshape(tf.transpose(xg), shape=[2500, 1])) +
                                             tf.square(Dy @ tf.reshape(tf.transpose(xg), shape=[2500, 1])))
                    qs_loss = tf.cast(qs_loss, dtype=tf.float32)

                    l1_loss = tf.reduce_mean(tf.abs(xg))
                    l1_loss = tf.cast(l1_loss, dtype=tf.float32)

                    critic_output = tf.reduce_mean(self.model(xg))

                    total_loss = data_loss + 0.5 * critic_output + 8 * qs_loss + 30 * l1_loss
                    total_loss = tf.multiply(total_loss, self.batch_size)

                    print(step, data_loss, critic_output)

                eval_gradients = eval_tape.gradient(total_loss, xg)[0]
                xg = xg - (step_size * eval_gradients)

                psnr_start = Metrics.psnr(np.asarray(xr), np.asarray(xg_start))
                psnr_current = Metrics.psnr(np.asarray(xr), np.asarray(xg[0, :, :, 0]))

            PlotUtils.plot_output(xr, xg_start, xg[0, :, :, 0], psnr_start, psnr_current)

            break


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
    experiment.log()
    experiment.train_regularizer(30)

    ind = 14
    for (gen_data, real_data, measurements) in experiment.train_dataset.take(1):
        output1 = experiment.model(gen_data[ind:ind+1, :, :])
        output2 = experiment.model(real_data[ind:ind+1, :, :])
        print("real data outputs: ", tf.reduce_mean(output2))
        print("gen data outputs: ", tf.reduce_mean(output1))
        break

    experiment.evaluate(ind, 20, 0.01)


import time
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
import numpy as np
import os
import trimesh

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    losses.mse(data, reconstruction))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def create_cnn():
    encoder_input = Input(shape=(2400, 3,))
    x = encoder_input

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(4, 3, activation='relu', padding='same')(x)
    x = Flatten()(x)

    x = Dense(16, activation='relu')(x)

    z_mean = layers.Dense(8, name="z_mean")(x)
    z_log_var = layers.Dense(8, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()


    latent_inputs = Input(shape=(8,))
    x = latent_inputs

    x = Dense(75 * 4, activation='relu')(x)
    x = Reshape((75, 4,))(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(3, 3, activation=None, padding='same')(x)
    decoded = x

    decoder = Model(latent_inputs, decoded, name="decoder")
    decoder.summary()

    autoencoder = VAE(encoder, decoder)
    autoencoder.compile(optimizer=optimizers.Adam())

    return autoencoder, encoder, decoder


def create_nn():
    encoder_input = Input(shape=(2319, 3,))
    x = encoder_input

    x = Flatten()(x)
    BatchNormalization()(x)

    x = Dense(1200, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(200, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(16, activation='relu')(x)

    z_mean = layers.Dense(8, name="z_mean")(x)
    z_log_var = layers.Dense(8, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(8,))
    x = latent_inputs

    x = Dense(200, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(1200, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(2319 * 3, activation=None)(x)
    BatchNormalization()(x)
    decoded = layers.Reshape((2319, 3))(x)

    decoder = Model(latent_inputs, decoded, name="decoder")
    decoder.summary()

    autoencoder = VAE(encoder, decoder)
    autoencoder.compile(optimizer=optimizers.Adam())

    return autoencoder, encoder, decoder



def create_cnn2():
    encoder_input = Input(shape=(2400, 3,))
    x = encoder_input

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D()(x)

    x = Conv1D(4, 3, activation='relu', padding='same')(x)
    x = Flatten()(x)

    x = Dense(16, activation='relu')(x)

    encoded = Dense(8, activation='relu')(x)
    encoder = Model(encoder_input, encoded, name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(8,))
    x = latent_inputs

    x = Dense(16, activation='relu')(x)

    x = Dense(75 * 4, activation='relu')(x)
    x = Reshape((75, 4,))(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D()(x)

    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = Conv1D(3, 3, activation=None, padding='same')(x)
    decoded = x

    decoder = Model(latent_inputs, decoded, name="decoder")
    decoder.summary()

    autoencoder = Model(encoder_input, decoder(encoder(encoder_input)), name="ae")
    autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.mse)
    autoencoder.summary()

    return autoencoder, encoder, decoder







def create_nn2():
    encoder_input = Input(shape=(22779, 3,))
    x = encoder_input

    x = Flatten()(x)
    BatchNormalization()(x)

    x = Dense(2400, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(400, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(200, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    encoded = Dense(8, activation='relu')(x)

    encoder = Model(encoder_input, encoded, name="encoder")
    encoder.summary()

    latent_inputs = Input(shape=(8,))
    x = latent_inputs

    x = Dense(200, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(400, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(2400, activation='relu')(x)
    BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(22779 * 3, activation=None)(x)
    BatchNormalization()(x)
    decoded = layers.Reshape((22779, 3))(x)

    decoder = Model(latent_inputs, decoded, name="decoder")
    decoder.summary()

    autoencoder = Model(encoder_input, decoder(encoder(encoder_input)), name="ae")
    autoencoder.compile(optimizer=optimizers.Adam(), loss=losses.mse)
    autoencoder.summary()

    return autoencoder, encoder, decoder


def load_data(path_, shape=(2319, 3), do_scale=True, subtract_mean=False):
    face_a = None
    data_set = []
    scale = 1.0

    for f in os.listdir(path_):
        mesh = trimesh.load(os.path.join(path_, f), process=False)
        v = np.array(mesh.vertices, np.float32)
        face_a = np.array(mesh.faces, np.int32)
        if shape == (2400, 3):
            v = np.vstack([v, np.zeros((81, 3), np.float32)])

        data_set.append(v)

    data_set = np.array(data_set)

    if do_scale:
        scale = np.max(data_set)
        data_set = data_set / scale

    mean_a = np.mean(data_set, axis=(0))

    if subtract_mean:
        data_set -= mean_a

    return data_set, face_a, mean_a, scale


def main(path_, train_=True):
    epochs = 2 ** 15
    batch_size = 16

    if train_:

        data_set, face_a, mean_a, scale = load_data(path_, shape=(22779, 3), do_scale=False, subtract_mean=False)
        vae, e, d = create_nn2()
        vae.fit(x=data_set, y=data_set, epochs=epochs, batch_size=batch_size, shuffle=True)
        e.save("./enc_nn2_noscale_nomean1/")
        d.save("./dec_nn2_noscale_nomean1/")

        """
        data_set, face_a, mean_a, scale = load_data(path_, shape=(2400, 3), do_scale=False, subtract_mean=False)
        vae, e, d = create_cnn2()
        vae.fit(x=data_set, y=data_set, epochs=epochs, batch_size=batch_size, shuffle=True)
        e.save("./enc_cnn2_scale_mean/")
        d.save("./dec_cnn2_scale_mean/")
        """








        """
        data_set, face_a, mean_a, scale = load_data(path_, shape=(2319, 3), do_scale=True, subtract_mean=False)
        vae, e, d = create_nn()
        vae.fit(data_set, epochs=epochs, batch_size=batch_size, shuffle=True)
        e.save("./enc_nn_scale_nomean/")
        d.save("./dec_nn_scale_nomean/")

        data_set, face_a, mean_a, scale = load_data(path_, shape=(2319, 3), do_scale=True, subtract_mean=True)
        vae, e, d = create_nn()
        vae.fit(data_set, epochs=epochs, batch_size=batch_size, shuffle=True)
        e.save("./enc_nn_scale_mean/")
        d.save("./dec_nn_scale_mean/")


        data_set, face_a, mean_a, scale = load_data(path_, shape=(2400, 3), do_scale=True, subtract_mean=False)
        vae, e, d = create_cnn()
        vae.fit(data_set, epochs=epochs, batch_size=batch_size, shuffle=True)
        e.save("./enc_cnn_scale_nomean/")
        d.save("./dec_cnn_scale_nomean/")

        data_set, face_a, mean_a, scale = load_data(path_, shape=(2400, 3), do_scale=True, subtract_mean=True)
        vae, e, d = create_cnn()
        vae.fit(data_set, epochs=epochs, batch_size=batch_size, shuffle=True)
        e.save("./enc_cnn_scale_mean/")
        d.save("./dec_cnn_scale_mean/")
        """


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    path_ = "../data/"
    start_time = time.time()
    main(path_)
    end_time = time.time()

    # convert from seconds into hours, minutes, seconds
    print(
        f"Runtime: {int((end_time - start_time) // 3600)}h "
        f"{int(((end_time - start_time) % 3600) // 60)}m "
        f"{int((end_time - start_time) % 60)}s."
    )

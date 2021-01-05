from __future__ import print_function, division
import scipy
from keras import metrics
# from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, LSTM, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing import sequence
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pandas as pd
# from data_loader import DataLoader
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class Universal():
    # A=R  B=T
    def __init__(self):
        # Input shape
        self.maccs_shape = (None, 166)  # (length, group)
        self.smile_shape = (None, 50)  # (length, char)
        self.x_shape = (None, 200)
        # Load data
        self.maccs_data = self.load_maccs_data()
        self.smile_data = self.load_smile_data()
        self.smile_data_3d = self.load_smile_data_3d()

        # set optimizer
        optimizer = Adam(1e-7, 0.5)  # (lr,beta)

        # Build the Dis_S
        self.Dis_S = self.build_Dis_S()
        self.Dis_S.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=['accuracy'])

        # Build and compile for Decoder
        self.D_X2M = self.build_D_X2M()
        self.D_X2S = self.build_D_X2S()

        # Build the Encoder
        self.E_M2X = self.build_E_M2X()
        self.E_S2X = self.build_E_S2X()
        # Build the Error_X
        self.X_error = self.build_X_error()
        # Build a coder for noise
        self.X_M_noise = self.build_X_M_noise()

        # Input images from both domains
        Maccs = Input(batch_shape=self.maccs_shape)
        Smile = Input(batch_shape=self.smile_shape)

        # Translate images to the other domain
        X_from_M = self.E_M2X(Maccs)
        X_from_S = self.E_S2X(Smile)
        X_from_M_with_noise = self.X_M_noise(self.E_M2X(Maccs))

        # Translate images back to original domain
        reconstr_M = self.D_X2M(X_from_M)
        generate_S = self.D_X2S(X_from_M_with_noise)
        reconstr_S = self.D_X2S(X_from_S)
        generate_M = self.D_X2M(X_from_S)
        X_error = self.X_error([X_from_M, X_from_S])

        # For the combined model we will only train the generators
        self.Dis_S.trainable = False

        # Discriminators determines validity of translated images
        discrmin_S = self.Dis_S(generate_S)

        # Combined model trains generators to fool discriminators
        self.AE_combined = Model(inputs=[Maccs, Smile],
                                 outputs=[reconstr_S,
                                          generate_M,
                                          reconstr_M,
                                          discrmin_S,
                                          X_error])
        self.AE_combined.compile(loss=['categorical_crossentropy', 'binary_crossentropy',
                                       'binary_crossentropy', 'binary_crossentropy',
                                       'mse'],
                                 loss_weights=[5, 1,
                                               1, 1,
                                               3],
                                 optimizer=optimizer,
                                 metrics=['accuracy'
                                          # metrics.binary_accuracy,
                                          # metrics.categorical_accuracy
                                          ]
                                 )

        self.Dis_S.summary()
        self.AE_combined.summary()
        self.AE_combined.save('./model_save/universal.h5')
        # self.save_generator()

    @staticmethod
    def load_maccs_data():
        a = pd.read_csv('./DataPool/maccsdata.csv', header=None).values
        return a

    @staticmethod
    def load_smile_data():
        b = pd.read_csv('./DataPool/smile.csv', header=None).values
        return b

    @staticmethod
    def load_smile_data_3d():
        smile_d = pd.read_csv('./DataPool/smile.csv', header=None).values
        target_texts = smile_d[:, :50]
        decoder_input_data = np.zeros((1303, 50, 161), dtype='float32')
        for i in range(1303):
            for k in range(50):
                decoder_input_data[i, k, int(target_texts[i, k])] = 1.
        return decoder_input_data

    def build_E_M2X(self):
        encoder_inputs = Input(batch_shape=self.maccs_shape)
        d0 = Dense(200, activation='relu')(encoder_inputs)
        output = Dense(200, activation='relu')(d0)
        return Model(encoder_inputs, output, name='E_M2X')

    def build_X_M_noise(self, latent_dim=200, epsilon_std=3.0):
        # noise part
        encoder_h = Input(batch_shape=self.x_shape)
        r0 = Reshape((200,))(encoder_h)
        # mean vector
        z_mean = Dense(latent_dim, activation='relu')(r0)
        # standard deviation vector
        z_log_var = Dense(latent_dim, activation='relu')(r0)

        def sampling(args):
            # Reparameterization trick
            z_mean, z_log_var = args
            # get epsilon from standard normal distribution
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        # latent vector z
        z = Lambda(sampling, output_shape=(None, latent_dim))([z_mean, z_log_var])
        return Model(encoder_h, z, name='X_M_noise')

    def build_D_X2S(self):
        # z  = X_M+noise = X_S
        z = Input(batch_shape=self.x_shape)
        z1 = Reshape((50, 4))(z)
        z1 = Dense(50, activation='relu')(z1)
        z1 = LSTM(161, return_sequences=True, return_state=False, activation='tanh')(z1)
        output = Dense(161, activation='softmax', name='output')(z1)
        return Model(z, output, name='D_X2S')

    def build_E_S2X(self):
        input_shape = Input(batch_shape=self.smile_shape)
        embedding_layer = Embedding(input_dim=160, input_length=50, output_dim=50, mask_zero=True)(input_shape)       # mask zero= true
        d0 = Flatten()(embedding_layer)
        output = Dense(units=200, activation='relu', name='E_S2X')(d0)
        return Model(input_shape, output, name='E_S2X')

    def build_D_X2M(self):
        # z  = X_M = X_S
        z = Input(batch_shape=self.x_shape)
        z1 = Reshape((200, -1))(z)
        L0 = LSTM(1000, return_sequences=False, return_state=False, activation='tanh')(z1)
        output = Dense(units=166, activation='hard_sigmoid', name='D_X2M')(L0)
        return Model(z, output, name='D_X2M')

    def build_Dis_S(self):
        d0 = Input(batch_shape=(None, 50, 161))
        d3 = Dense(128, activation='relu')(d0)
        d4 = Dense(64, activation='relu')(d3)
        d4 = Flatten()(d4)
        output = Dense(1, activation='hard_sigmoid', name='Dis_0_1_layer')(d4)
        return Model(d0, output, name='Dis_S')

    def build_X_error(self):
        x = Input(batch_shape=self.x_shape, name='x_input')
        x_ = Input(batch_shape=self.x_shape, name='x__input')
        output = Lambda(lambda e: K.square(e[0] - e[1]))([x, x_])
        return Model([x, x_], output, name='x_error')

    def data_loader(self, batch_size):
        dataA = self.load_maccs_data()
        dataB = self.load_smile_data()
        dataC = self.load_smile_data_3d()
        self.n_batch = int(min(len(dataA), len(dataB)) / batch_size)   # self.n_batch=20 = 1303 / 64
        total_samples = self.n_batch * batch_size
        for i in range(self.n_batch):
            batchA = dataA[i * batch_size:(i + 1) * batch_size, :]
            batchB = dataB[i * batch_size:(i + 1) * batch_size, :]
            batchC = dataC[i * batch_size:(i + 1) * batch_size, :]
            yield batchA, batchB, batchC

# this part don't work
    @staticmethod
    def check_fake_smile(gen_smile):
        smiles = pd.read_csv('./DataPool/smile.csv', header=None).values
        for i in range(len(smiles)):
            if gen_smile == smiles[i]:
                return None
            else:
                return gen_smile


    def train(self, epochs, batch_size=1):
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        # Train the generators
        for epoch in range(epochs):
            self.R = epoch
            self.T = epochs
            for batch_i, (self.maccs_data, self.smile_data, self.smile_data_3d) in enumerate(
                    self.data_loader(batch_size=batch_size)):
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # discriminator training part
                fake_smile = self.check_fake_smile(self.D_X2S.predict(self.E_M2X.predict(self.maccs_data)))
                d_loss_real = self.Dis_S.train_on_batch(self.smile_data_3d, valid)  # valid 1
                d_loss_fake = self.Dis_S.train_on_batch(fake_smile, fake)  # fake 0
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                maccs_train, maccs_test, smile_train, smile_test, smile_train_3d, smile_test_3d = train_test_split \
                    (self.maccs_data, self.smile_data, self.smile_data_3d, test_size=0.2, random_state=33)

                # ------------------
                #  Train Generators
                # ------------------

                # generator training part
                '''        self.AE_combined = Model(inputs=[Maccs, Smile],
                                 outputs=[reconstr_S, generate_M,
                                          reconstr_M, discrmin_S,
                                          X_error])'''
                g_loss = self.AE_combined.train_on_batch([maccs_train, smile_train],
                                                         [smile_train_3d,
                                                          maccs_train,
                                                          maccs_train,
                                                          np.ones((len(maccs_train), 1), dtype='float32'),
                                                          np.zeros((len(maccs_train), 200), dtype='float32')])

                print(
                    "[Epoch %d/%d] [Batch %d/%d] "
                    "[D loss: %f, acc: %3d%%,%3d%%,%3d%% ] "
                    "[G loss: %05f, "
                    "rec_S: %05f,%05f, "
                    "gen_M: %05f,%05f, "
                    "rec_M: %05f,%05f,  "
                    "dis_S: %05f,%05f,  "
                    "X_error: %05f, ]"
                    \
                    % (epoch + 1, epochs, batch_i + 1, self.n_batch,
                       d_loss[0], 100 * d_loss[1], 100 * d_loss_real[1], 100 * d_loss_fake[1],
                       g_loss[0],
                       g_loss[1], g_loss[6],  # categorical
                       g_loss[2], g_loss[7],  # binary
                       g_loss[3], g_loss[8],  # binary
                       g_loss[4], g_loss[9],  # binary
                       g_loss[5],  # loss
                       ))

        # Save models
        self.save_models()

    def save_models(self):
        os.makedirs('h5/G/G_%sin%s' % (self.R, self.T), exist_ok=True)
        self.AE_combined.save('./h5/G/G_%sin%s/g_%sin%s.h5' % (self.R, self.T, self.R, self.T))

    def load_model(self):

        self.AE_combined = load_model('./h5/G/G_%s%s/g_%s%s.h5' % (self.R, self.T, self.R, self.T))

    def M2S_test(self):
        return 0

    def M2M_test(self):
        return 0

    def S2M_test(self):
        return 0

    def S2S_test(self):
        return 0

    def X_error_test(self):
        return 0


# 如果在其他的py檔想執行universal 那以下單元測試也會被執行 可以把if判斷改成false就不會執行(how?)
if __name__ == '__main__':
    uni = Universal()  # load data build model
    uni.train(epochs=300, batch_size=64)
    # uni.S2M_test()
    # uni.S2S_test()


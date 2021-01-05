import numpy as np
import pandas as pd
from keras import metrics
from keras.layers import Input, Dense, Lambda, Layer, Embedding, LSTM, Reshape, GRU
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard

# select data
maccs_train = pd.read_csv('./DataPool/select data/maccs_train.csv', index_col=None, header=None).values
maccs_test = pd.read_csv('./DataPool/select data/maccs_test.csv', index_col=None, header=None).values
smile_train = pd.read_csv('./DataPool/select data/smile_train_with_161162.csv', index_col=None, header=None).values
smile_test = pd.read_csv('./DataPool/select data/smile_test_with_161162.csv', index_col=None, header=None).values
compound_train0 = pd.read_csv('./DataPool/select data/compound_train.csv', index_col=None, header=None)
compound_test0 = pd.read_csv('./DataPool/select data/compound_test.csv', index_col=None, header=None)
input_texts = maccs_train[:, :166]
xtest = maccs_test[:, :166]
target_texts = smile_train[:, :52]
ytest = smile_test[:, :52]
compound_train = list(compound_train0[0])
compound_test = list(compound_test0[0])


encoder_input_data = input_texts.reshape(1097, 166, 1)
decoder_input_data = np.zeros((1097, 52, 163), dtype='float32')
decoder_target_data = np.zeros((1097, 52, 163), dtype='float32')
for i in range(1097):
    for k in range(52):
        decoder_input_data[i, k, int(target_texts[i, k])] = 1.
        if k > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, k - 1, int(target_texts[i, k])] = 1.


adata = xtest.reshape(275, 166, 1)
bdata = np.zeros((275, 52, 163), dtype='float32')
cdata = np.zeros((275, 52, 163), dtype='float32')
for i in range(275):
    for k in range(52):
        bdata[i, k, int(ytest[i, k])] = 1.
        if k > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            cdata[i, k - 1, int(ytest[i, k])] = 1.

encoder_inputs = Input(shape=(166, 1))
encoder_outputs, state_h, state_c = LSTM(332, return_state=True)(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.


def sampling(args):
    # Reparameterization trick
    [z_mean, z_log_var] = args
    # get epsilon from standard normal distribution
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 332), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# mean vector
z_mean = Dense(332)(state_h)
z_mean_c = Dense(332)(state_c)
# standard deviation vector
z_log_var = Dense(332)(state_h)
z_log_var_c = Dense(332)(state_c)
# latent vector z
tate_h = Lambda(sampling, output_shape=(332,))([z_mean, z_log_var])
tate_c = Lambda(sampling, output_shape=(332,))([z_mean_c, z_log_var_c])
encoder_states = [tate_h, tate_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 163))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(332, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(163, activation='softmax', name='output')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# Callbacks
visualization_callback = TensorBoard(log_dir='./123logs_M2S_S2S_noise_v3/', write_graph=True, write_images=True)
EarlyStopping_callback = EarlyStopping(patience=100, monitor='loss',
                                       mode='min')  # patience 沒有進步的訓練輪數 之後訓練就會被停止 (100次之後沒進步就會停)
CSVLogger_callback = CSVLogger('./logs.csv')
weight_save_callback = ModelCheckpoint('./model_save/123M2S_S2S_noise_v3.h5', monitor='loss',
                                       verbose=0,
                                       save_best_only=True, mode='min', period=1)  # 在每個訓練期之後保存模型
# Run training
model.compile(optimizer='Adam', loss='mse', metrics=[metrics.categorical_accuracy])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=5000,
          # validation_split=0.1,
          validation_data=([adata[:64], bdata[:64]], cdata[:64]),
          shuffle=True, callbacks=[CSVLogger_callback, weight_save_callback, EarlyStopping_callback,
                                   visualization_callback])
# Save model
model.save('./model_save/123M2S_S2S_noise_v3.h5')


Dense_model = load_model('./model_save/123M2S_S2S_noise_v3.h5')
loss = Dense_model.evaluate([adata, bdata], cdata, batch_size=2000)
print('\ntest loss:', loss)


z_mean = Model(encoder_inputs, z_mean)
z_mean_c = Model(encoder_inputs, z_mean_c)
z_log_var = Model(encoder_inputs, z_log_var)
z_log_var_c = Model(encoder_inputs, z_log_var_c)

decoder_state_input_h = Input(shape=(332,))
decoder_state_input_c = Input(shape=(332,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# z_mean.save('M2S_S2S_noise_z_mean_v3.h5')
# z_mean_c.save('M2S_S2S_noise_z_mean_c_v3.h5')
# z_log_var.save('M2S_S2S_noise_z_log_var_v3.h5')
# z_log_var_c.save('M2S_S2S_noise_z_log_var_c_v3.h5')
#
# decoder_model.save('M2S_S2S_noise_decoder_v3.h5')







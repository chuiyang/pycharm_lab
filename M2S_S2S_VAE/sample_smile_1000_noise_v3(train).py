import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import csv

# select data
maccs_train = pd.read_csv('./DataPool/select data/maccs_train.csv', index_col=None, header=None).values
maccs_test = pd.read_csv('./DataPool/select data/maccs_test.csv', index_col=None, header=None).values
smile_train_v = pd.read_csv('./DataPool/select data/smiles_train_v.csv', index_col=None, header=None).values[:, :50]


input_texts = maccs_train[:, :166]
xtest = maccs_test[:, :166]


encoder_input_data = input_texts.reshape(1097, 166, 1)
adata = xtest.reshape(275, 166, 1)

# for b in range(1):
b = 208
index = b + 1
amount = 1000
std = 4.0

# encoder output
a = load_model('./model_save/M2S_S2S_noise_z_mean_v3.h5')
e = load_model('./model_save/M2S_S2S_noise_z_mean_c_v3.h5')
c = load_model('./model_save/M2S_S2S_noise_z_log_var_v3.h5')
d = load_model('./model_save/M2S_S2S_noise_z_log_var_c_v3.h5')
decoder = load_model('./model_save/M2S_S2S_noise_decoder_v3.h5')

z_mean = a.predict(np.reshape(encoder_input_data[b], (1, 166, 1)))
z_mean_c = e.predict(np.reshape(encoder_input_data[b], (1, 166, 1)))
z_log_var = c.predict(np.reshape(encoder_input_data[b], (1, 166, 1)))
z_log_var_c = d.predict(np.reshape(encoder_input_data[b], (1, 166, 1)))

epsilon = np.random.normal(loc=0.0, scale=std, size=amount)
sess = tf.Session()
output_1000 = np.zeros((amount, 50), dtype='float32')

with open(str(index) + '_maccs-smile_' + str(amount) + '(std='+str(std) + ').v3.csv', 'w', newline='') as output:
    writer = csv.writer(output)
    yes = 0
    number = []
    for i in range(amount):
        z_ = z_mean + K.exp(z_log_var / 2) * epsilon[i]
        z = z_.eval(session=sess)
        z_c_ = z_mean_c + K.exp(z_log_var_c / 2) * epsilon[i]
        z_c = z_c_.eval(session=sess)
        states_value = [z, z_c]
        target_seq = np.zeros((1, 1, 163))
        target_seq[0, 0, 161] = 1.
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = decoder.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_sentence.append(sampled_token_index)

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_token_index == 162 or
               len(decoded_sentence) > 52):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 163))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        # print(decoded_sentence)
        if 162 in decoded_sentence:
            decoded_sentence.remove(162)
        # print(decoded_sentence)
        to_array = np.array(decoded_sentence).reshape(1, len(decoded_sentence))
        smile_generated = pad_sequences(to_array, maxlen=50, dtype='int32', padding='post', truncating='post', value=0.)
        # print(smile_generated)
        writer.writerows(smile_generated)
        if np.array_equal(smile_generated[0], smile_train_v[b]):
            yes += 1
            number.append(i)

    print(yes)
    print(number)
    print(smile_train_v[b])








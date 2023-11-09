import tensorflow as tf
import tensorflow.nn as nn
import keras
from keras.layers import Input, Dense, Lambda
from keras import initializers
from keras.regularizers import l2
import numpy as np
from keras import Model
from Reader.Reader import Reader
from scipy.sparse import csr_matrix
from keras.layers import LeakyReLU


def get_model(input_len, layers=[512, 256, 64]):
    input = Input(shape=(input_len, ), dtype=tf.float32, name='input')
    x = Dense(layers[0], activation='relu', name='encode0',kernel_initializer=initializers.lecun_normal())(input)
    for i in range(1, len(layers)-1):
        x = Dense(layers[i], activation='relu', name='encode%d'%i, kernel_initializer=initializers.lecun_normal())(x)
    code = Dense(layers[-1], activation='sigmoid')(x)
    decode_ind = list(np.linspace(len(layers)-2, 0, len(layers)-1).astype(int))
    for i in decode_ind:
        x = Dense(layers[i], activation='relu', name='dencode%d'%i, kernel_initializer=initializers.lecun_normal())(x)
    output = Dense(input_len, kernel_initializer=initializers.lecun_normal())(x)
    model = Model(inputs=input, outputs=output)
    encoder = Model(input, code)
    return model, encoder


def myloss(y_true, y_pre, alpha=1.0):
    y_pre = tf.where(tf.equal(y_true, tf.zeros_like(y_true)), tf.zeros_like(y_true), y_pre)
    conut_nozero = tf.math.count_nonzero(y_true, axis=1)
    #weights = alpha * (y_pre.shape[1] // conut_nozero)//y_pre.shape[1]
    weights = alpha / tf.cast(conut_nozero, tf.float32)
    error = tf.reduce_sum(tf.square(y_pre - y_true),axis=1)
    error = tf.divide(error, tf.cast(conut_nozero, tf.float32))
    weights = tf.cast(tf.reshape(weights,[-1,1]), tf.float32)
    error = tf.cast(tf.reshape(error, [-1,1]), tf.float32)
    loss = tf.multiply(weights, error)
    loss = tf.reduce_mean(loss)
    return loss


if __name__ == '__main__':
    reader = Reader()
    u_row = []
    u_col = []
    ratings = []
    for u in reader.dataSet_u:
        for i in reader.dataSet_u[u]:
            u_row.append(reader.user[u])
            u_col.append(reader.item[i])
            ratings.append(reader.dataSet_u[u][i])
    u_mat = csr_matrix((ratings, (u_row, u_col)), shape=(len(reader.user), len(reader.item))).toarray()
    i_mat = u_mat.transpose()

    print('shape:'+str(u_mat.shape))
    # [512, 128, 64]
    model, encoder = get_model(len(u_mat[0]),layers=[512,128,64])
    model.compile(loss=myloss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['mse'])
    #print(encoder.predict(u_mat[0].reshape(1, -1)))
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')]
    model.fit(u_mat, u_mat, validation_split=0.1, epochs=30, batch_size=256, callbacks=callbacks, verbose=1, shuffle=True)
    u_mat = encoder.predict(u_mat)
    #print(u_mat.shape)
    np.savetxt('../data/%s/split/dae_u.txt'%reader.config.dataset_name, u_mat, delimiter=' ', fmt='%.3f')
    print(len(i_mat[0]))
    # [512, 128, 32]
    model, encoder = get_model(len(i_mat[0]), layers=[512,128,32])
    model.compile(loss=myloss, optimizer=keras.optimizers.Adam(lr=0.001), metrics=['mse'])
    #print(encoder.predict(i_mat[0].reshape(1, -1)))
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')]
    model.fit(i_mat, i_mat, validation_split=0.1, epochs=30, batch_size=256, callbacks=callbacks, verbose=1,
              shuffle=True)
    i_mat = encoder.predict(i_mat)
    #print(i_mat.shape)
    np.savetxt('../data/%s/split/dae_i.txt'%reader.config.dataset_name, i_mat, delimiter=' ', fmt='%.3f')
    #print(encoder.predict(u_mat[0].reshape(1, -1)))
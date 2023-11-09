import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from Reader.config import config
from keras.initializers import TruncatedNormal
import numpy as np
from keras.layers import Input, Lambda, Reshape, Dense, Multiply, Flatten, Embedding, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
from Reader.Reader import Reader

# layers=[256, 128, 64], userlayers=[128, 128, 64], itemlayers=[256, 128, 64], num_uf=64, num_if=32

def get_model(num_users, num_items, layers=[256, 128, 64], userlayers=[128, 128, 64], itemlayers=[256, 128, 64], num_uf=64, num_if=32):
    conf = config()
    u_ratings = tf.constant(np.loadtxt('../data/%s/split/dae_u.txt' % conf.dataset_name, delimiter=' '))
    u_ratings = tf.cast(u_ratings, tf.float32)
    i_ratings = tf.constant(np.loadtxt('../data/%s/split/dae_i.txt' % conf.dataset_name, delimiter=' '))
    i_ratings = tf.cast(i_ratings, tf.float32)
    user_input = Input(shape=(1,) , dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    num_layer = len(layers)
    dmf_len = len(userlayers)

    user_ratings = Lambda(lambda x: tf.gather(u_ratings, x))(user_input)
    item_ratings = Lambda(lambda x: tf.gather(i_ratings, x))(item_input)
    user_ratings = Reshape((num_uf,))(user_ratings)
    item_ratings = Reshape((num_if,))(item_ratings)

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=userlayers[0], name='mf_embedding_user',
                                  embeddings_regularizer=l2(0.006), input_length=1)(user_input)
    MF_Embedding_User = Reshape((userlayers[0],))(MF_Embedding_User)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=itemlayers[0], name='mf_embedding_item',
                                  embeddings_regularizer=l2(0.006), input_length=1)(item_input)
    MF_Embedding_Item = Reshape((itemlayers[0],))(MF_Embedding_Item)
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   embeddings_regularizer=l2(0.006), input_length=1)(user_input)
    MLP_Embedding_User = Reshape((layers[0]//2, ))(MLP_Embedding_User)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_regularizer=l2(0.006), input_length=1)(item_input)
    MLP_Embedding_Item = Reshape((layers[0]//2,))(MLP_Embedding_Item)

    #DMF part
    for idx in range(1, len(userlayers)):
        MF_Embedding_User = Dense(userlayers[idx], activation='relu', name='dmf_user_layer%d' % idx,
                                  kernel_regularizer=l2(0.01))(MF_Embedding_User)
    for idx in range(1, len(itemlayers)):
        MF_Embedding_Item = Dense(itemlayers[idx], activation='relu', name='dmf_item_layer%d' % idx,
                                  kernel_regularizer=l2(0.01))(MF_Embedding_Item)
    dmf_vector = Multiply()([MF_Embedding_User, MF_Embedding_Item])
    # MLP part
    mlp_vector = concatenate([MLP_Embedding_User, MLP_Embedding_Item])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(0.01), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    item_ratings = Dense(num_if, activation='relu', kernel_initializer='lecun_uniform')(item_ratings)
    item_ratings = Dense(num_if//2, activation='relu', kernel_initializer='lecun_uniform')(item_ratings)
    #item_ratings = Dense(num_if//4, activation='relu', kernel_regularizer=l2(0.002), kernel_initializer='lecun_uniform')(item_ratings)
    user_ratings = Dense(num_uf, activation='relu', kernel_initializer='lecun_uniform')(user_ratings)
    user_ratings = Dense(num_uf // 2, activation='relu', kernel_initializer='lecun_uniform')(user_ratings)
    #user_ratings = Dense(num_uf // 4, activation='relu', kernel_regularizer=l2(0.002), kernel_initializer='lecun_uniform')(user_ratings)
    #
    golbal_features = concatenate([user_ratings, item_ratings])
    predict_vector = concatenate([dmf_vector, mlp_vector, golbal_features])
    #predict_vector = concatenate([dmf_vector, mlp_vector])
    #predict_vector = Dense(32, kernel_initializer='lecun_uniform', name='prediction_mlp')(predict_vector)
    prediction = Dense(1, kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

    model = keras.Model(inputs=[user_input, item_input], outputs=prediction)

    return model



def getTrainData():
    conf = config()
    trian_path = conf.train_path
    user_input, item_input, labels = [], [], []
    with open(trian_path, 'r') as f:
        for line in f.readlines():
            u, i, r = line.strip('\r\n').split(' ')
            user_input.append(int(u))
            item_input.append(int(i))
            labels.append(float(r))
    return user_input, item_input, labels


def evaluate_model(model, u2id, i2id):
    conf = config()
    test_path = conf.test_path
    user_id = []
    item_id = []
    ratings = []
    prediction = []
    with open(test_path, 'r') as f:
        for line in f.readlines():
            u, i, r = line.split(' ')
            user_id.append(u2id[int(u)])
            item_id.append(i2id[int(i)])
            ratings.append(float(r))

    pred = model.predict([np.array(user_id).reshape(-1,1), np.array(item_id).reshape(-1,1)], batch_size=512)
    mae = sum(np.abs(np.array(ratings) - pred.flatten()))/len(pred)
    rmse =np.sqrt(sum(np.square(np.array(ratings) - pred.flatten()))/len(pred))
    return rmse, mae



if __name__ == '__main__':
    data_reader = Reader()
    user_input, item_input, labels = getTrainData()
    user_input = [data_reader.user[i] for i in user_input]
    item_input = [data_reader.item[i] for i in item_input]
    num_users = max(user_input) + 1
    print(num_users)
    num_items = max(item_input) + 1
    print(num_items)
    model = get_model(num_users, num_items)
    print(model.summary())
    model.compile(optimizer=Adam(lr=0.001), loss='mse')


    rmse, mae = evaluate_model(model, data_reader.user, data_reader.item)
    best_rmse, best_mae, best_iter = rmse, mae, -1
    print('Iteration -1: rmse = %.4f, mae = %.4f'
          % (rmse, mae))
    for epoch in range(100):
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                  np.array(labels),  # labels
                  batch_size=1024, epochs=1, verbose=0, shuffle=True)
        if epoch % 1 == 0:
            (rmses, maes) = evaluate_model(model, data_reader.user, data_reader.item)
            rmse, mae, loss = rmses, maes, hist.history['loss'][0]
            # print(rmse)
            # print(mae)
            # print(loss)
            print('Iteration %d: rmse = %.4f, mae = %.4f, loss = %.4f'
                  % (epoch, rmse, mae, loss))
            if rmse < best_rmse:
                best_rmse, best_mae, best_iter = rmse, mae, epoch
            # else:
            #     break
    print('Iteration %d: rmse = %.4f, mae = %.4f'
          % (best_iter, best_rmse, best_mae))



# -*- coding: utf-8 -*-

from hpara_select.data_proc import *
from hpara_select.cnn_tf2 import *
from hpara_select.dnn_tf2 import *
from hpara_select.lstm_tf2 import *
from hpara_select.word2vec import *

if __name__ == '__main__':

    vx_train = np.load(x_train_dir)
    vx_test = np.load(x_test_dir)
    vy_train = np.load(y_train_dir)
    vy_test = np.load(y_test_dir)

    # vy_test = np.array(vy_test)
    # vy_train = np.array(vy_train)

    # onehot编码器

    values = np.array([0, 1])
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    vy_train = onehot_encoder.transform(vy_train.reshape(-1,
                                                         1))  # 对y进行onehot编码
    vy_test = onehot_encoder.transform(vy_test.reshape(-1, 1))  # 对y进行onehot编码

    x_test = vx_test
    y_test = vy_test

    x_train = vx_train
    y_train = vy_train

    x_test = np.array(x_test).reshape(-1, time_step *
                                      embedding_size+low_feature_dim)
    y_test = np.array(y_test).reshape(-1, 2)
    x_train = np.array(x_train).reshape(-1, time_step *
                                        embedding_size+low_feature_dim)
    y_train = np.array(y_train).reshape(-1, 2)

    for tmp_n_inputs in [30, 50, 100, 150, 200, 250, 300, 350 ,400]:

        train_lstm(
            x_train,
            x_test,
            y_train,
            y_test,
            tmp_n_inputs,
            lr=0.001,
            BATCH_SIZE=1024,
            training_iters=1000000,
            # training_iters=1000,
            write_result_to_file=True,
            result_file='result/hpara_select_lstm_'+str(tmp_n_inputs)+'.txt')

    for tmp_n_inputs in [30, 50, 100, 150, 200, 250, 300, 350 ,400]:

        train_cnn(
            x_train,
            x_test,
            y_train,
            y_test,
            tmp_n_inputs,
            lr=0.001,
            BATCH_SIZE=1024,
            training_iters=1000000,
            # training_iters=1000,
            write_result_to_file=True,
            result_file='result/hpara_select_cnn_'+str(tmp_n_inputs)+'.txt')

    for tmp_n_inputs in [30, 50, 100, 150, 200, 250, 300, 350 ,400]:

        train_dnn(
            x_train,
            x_test,
            y_train,
            y_test,
            tmp_n_inputs,
            lr=0.001,
            BATCH_SIZE=1024,
            training_iters=1000000,
            # training_iters=1000,
            write_result_to_file=True,
            result_file='result/hpara_select_dnn_'+str(tmp_n_inputs)+'.txt')

# coding=utf-8

from utils import *
from discriminator.word2vec import *
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import argmax
from vectorize import vectorize_ob_dataset_from_dir


def eval_brief(y_test, y_pred, pre_wr, result_file):
    with open(result_file, 'a') as fw:
        print(pre_wr, file=fw)
        print('Test acc:', accuracy_score(y_test, y_pred), file=fw)
        print('Test recall:', recall_score(y_test, y_pred), file=fw)
        print('Test precision:', precision_score(y_test, y_pred), file=fw)
        print('Test f1-score:', f1_score(y_test, y_pred), file=fw)
        print('***************', file=fw)


def eval_model(model, x_test, y_test, result_name, result_file):
    '''
    对模型model在x_test上进行评估，写入result_file文件中；result_name为写入前加的说明语句
    '''
    y_pred = model.predict(x_test, batch_size=x_test.shape[0])

    y_pred_real = []
    for i in range(y_pred.shape[0]):
        y_pred_real.append(label_encoder.inverse_transform([argmax(y_pred[i])
                                                            ]))

    eval_brief(y_test, y_pred_real, result_name+': ', result_file)


values = np.array([0, 1])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)


if __name__ == '__main__':

    data_li = [
        'att/',
        'lstm/',
        'cnn/',
        'dnn/'
    ]

    num_dir = 'result/ob_shell_num_all.txt'

    fw = open(num_dir, 'w')  # 清空记录文件
    fw.close()

    fw = open('result/ob_all_test.txt', 'w')  # 清空记录文件
    fw.close()

    from discriminator.attention_tf2 import build_model
    att = build_model()
    att.load_weights('model/attention_weights.keras')

    from discriminator.attention_tf2 import build_model
    att_gan = build_model()
    att_gan.load_weights('model/attention_weights_7.keras')

    from discriminator import lstm_tf2
    lstm = lstm_tf2.build_model()
    lstm.load_weights('model/lstm_weights.keras')

    from discriminator import lstm_tf2
    lstm_gan = lstm_tf2.build_model()
    lstm_gan.load_weights('model/lstm_weights_7.keras')

    from discriminator import cnn_tf2
    cnn = cnn_tf2.build_model()
    cnn.load_weights('model/cnn_weights.keras')

    from discriminator import cnn_tf2
    cnn_gan = cnn_tf2.build_model()
    cnn_gan.load_weights('model/cnn_weights_7.keras')

    from discriminator import dnn_tf2
    dnn = dnn_tf2.build_model()
    dnn.load_weights('model/dnn_weights.keras')

    from discriminator import dnn_tf2
    dnn_gan = dnn_tf2.build_model()
    dnn_gan.load_weights('model/dnn_weights_7.keras')

    model_li = {
        'att': att,
        'att_gan': att_gan,
        'lstm': lstm,
        'lstm_gan': lstm_gan,
        'cnn': cnn,
        'cnn_gan': cnn_gan,
        'dnn': dnn,
        'dnn_gan': dnn_gan
    }

    x_npy_dir = 'bin/x_obfuscated_all.npy'
    y_npy_dir = 'bin/y_obfuscated_all.npy'

    for data_name in data_li:

        data_dir = 'data/obfuscated_bak_'+data_name

        # 向量化混淆的webshell为numpy

        vectorize_ob_dataset_from_dir(data_dir, x_npy_dir,
                                      y_npy_dir,)

        x_obfuscated = np.load(x_npy_dir)
        y_obfuscated = np.load(y_npy_dir)

        with open(num_dir, 'a') as fw:
            print('[num] 本轮混淆shell数据量为：', len(y_obfuscated), file=fw)

        x_test = x_obfuscated.reshape(-1, time_step *
                                      embedding_size+low_feature_dim)
        y_test = y_obfuscated

        fw = open(num_dir, 'a')
        fw.write(data_name[:-1]+'：')
        fw.write(str(y_obfuscated.shape[0])+'\n')
        fw.close()

        for model_name in list(model_li.keys()):
            eval_model(model_li[model_name], x_test, y_test, 'model: {};data: {}'.format(
                model_name, data_dir), 'result/ob_all_test.txt')

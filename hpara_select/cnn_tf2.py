from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
from utils import *

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model

# from imblearn.over_sampling import RandomOverSampler as ros

# you may need this for TF-gpu under v2.4
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


heat_map_dir = 'result/cnn.png'  # 热图路径

n_inputs = embedding_size  # 输入维度，等于embedding大小
n_steps = time_step
# n_hidden_units = 50  # 隐藏层层数（不一定等于时序长度）

# --参数区
n_classes = 2
# input_keep_prob = 0.6  # dropout层
CONTINUE_TRAIN = True


# 超参数
lr = 0.01
training_iters = 1000000  # 迭代次数（不是epoch数） epoch=training_iters/data_len

random_state = 0
tf.random.set_seed(0)


def build_model(n_hidden_units, lr=0.001):  # 构建cnn模型
    inputs_all = tf.keras.Input(shape=(
        n_steps*n_inputs+low_feature_dim
    ))

    inputs = tf.keras.layers.Lambda(
        lambda x: x[:, :-low_feature_dim])(inputs_all)

    inputs = layers.Reshape((n_steps, n_inputs))(inputs)

    inputs_low = tf.keras.layers.Lambda(
        lambda x: x[:, -low_feature_dim:])(inputs_all)

    t_inputs = tf.reshape(inputs, (-1, n_steps, n_inputs, 1))

    h1 = layers.Convolution2D(
        batch_input_shape=(None, n_steps,  n_inputs, 1),
        filters=1,
        kernel_size=5,
        strides=1,
        padding='same',
        data_format='channels_last'
    )(t_inputs)

    h2 = layers.Activation('relu')(h1)

    h3 = layers.MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same'
    )(h2)

    h4 = layers.Flatten()(h3)  # 展开后使用全连接

    h4_con = layers.Concatenate()([h4, inputs_low])  # 拼接低维特征

    h5 = layers.Dense(n_classes, input_shape=(
        time_step*n_hidden_units,))(h4_con)

    outputs = layers.Activation('softmax')(h5)

    model = tf.keras.Model(inputs=inputs_all, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def conf_max(y, y_pred, name_li):

    names = name_li
    labels = [i for i in range(len(names))]
    conf_mat = confusion_matrix(y, y_pred, labels=labels)

    df_conf_mat = pd.DataFrame(conf_mat)

    index = [i for i in range(len(names))]
    index_dic = zip(index, names)
    index_dic = dict(index_dic)
    # print(index_dic)
    df_conf_mat = df_conf_mat.rename(index=index_dic, columns=index_dic)
    mymap = seaborn.heatmap(df_conf_mat, annot=True)
    plt.savefig(heat_map_dir)


def train_cnn(
        vx_train,
        vx_test,
        vy_train,
        vy_test,
        n_hidden_units,
        lr=lr,
        training_iters=training_iters,
        BATCH_SIZE=BATCH_SIZE,  # 128 训练时批的大小
        write_result_to_file=False,
        result_file='result/imbalance.txt',
        IS_GAN=False,
        GAN_DIR='model/cnn_weights.keras'
    # VAL_SIZE=VAL_SIZE,
):
    '''
    训练cnn神经网络
    '''
    train_model = build_model(n_hidden_units, lr=lr)

    # label编码器
    values = np.array([0, 1])
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(values)

    x_train, x_test, y_train, y_test = vx_train, vx_test, vy_train, vy_test

    x_train = x_train.reshape(-1, time_step * embedding_size+low_feature_dim)

    if x_train.shape[0] <= BATCH_SIZE:

        train_model.fit(x_train,
                        y_train,
                        batch_size=x_train.shape[0],
                        epochs=training_iters // x_train.shape[0],
                        shuffle=True)

    else:
        # 使数据量能够整除batch_size
        data_num = x_train.shape[0]
        cut_num = data_num % BATCH_SIZE
        if cut_num != 0:
            x_train = x_train[:-cut_num]
            y_train = y_train[:-cut_num]

        train_model.fit(x_train,
                        y_train,
                        batch_size=BATCH_SIZE,
                        epochs=training_iters // x_train.shape[0],
                        shuffle=True)

    # y_pred = train_model.predict(x_test, batch_size=x_test.shape[0])
    y_pred = train_model.predict(x_test)

    y_pred_real = []
    for i in range(y_pred.shape[0]):
        y_pred_real.append(label_encoder.inverse_transform([argmax(y_pred[i])
                                                            ]))

    y_test_real = []
    for i in range(y_test.shape[0]):
        y_test_real.append(label_encoder.inverse_transform([argmax(
            y_test[i])]))  # 将y_test进行onehot还原

    if write_result_to_file:
        with open(result_file, 'a') as fw:
            print('Test acc:' + str(accuracy_score(y_test_real, y_pred_real)),
                  file=fw)
            print('Test recall:' + str(recall_score(y_test_real, y_pred_real)),
                  file=fw)
            print('Test precision:' +
                  str(precision_score(y_test_real, y_pred_real)),
                  file=fw)
            print('Test f1-score:' + str(f1_score(y_test_real, y_pred_real)),
                  file=fw)
            print('*******************', file=fw)


def eval_result(x_test,
                y_test,
                model_save_dir='model/cnn_weights.keras',
                if_conf_max=True,
                write_to_file=False,
                result_file='result/train.txt'):
    '''
    评估训练结果，
    输入：x_test,y_test,model_dir
    会print评估结果、保存混淆矩阵
    '''

    # label编码器
    values = np.array([0, 1])
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(values)

    eval_model = build_model()
    eval_model.load_weights(model_save_dir)

    # y_pred = eval_model.predict(x_test, batch_size=x_test.shape[0])
    y_pred = eval_model.predict(x_test)

    y_pred_real = []
    for i in range(y_pred.shape[0]):
        y_pred_real.append(label_encoder.inverse_transform([argmax(y_pred[i])
                                                            ]))

    y_test_real = []
    for i in range(y_test.shape[0]):
        y_test_real.append(label_encoder.inverse_transform([argmax(
            y_test[i])]))  # 将y_test进行onehot还原

    print()
    print('Test acc:', accuracy_score(y_test_real, y_pred_real))
    print('Test recall:', recall_score(y_test_real, y_pred_real))
    print('Test precision:', precision_score(y_test_real, y_pred_real))
    print('Test f1-score:', f1_score(y_test_real, y_pred_real))

    if if_conf_max:
        pass
        # conf_max(y_test_real, y_pred_real)

    if write_to_file:
        with open(result_file, 'a') as fw:
            print('Test acc:',
                  accuracy_score(y_test_real, y_pred_real),
                  file=fw)
            print('Test recall:',
                  recall_score(y_test_real, y_pred_real),
                  file=fw)
            print('Test precision:',
                  precision_score(y_test_real, y_pred_real),
                  file=fw)
            print('Test f1-score:',
                  f1_score(y_test_real, y_pred_real),
                  file=fw)

# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from utils import *

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import seaborn
import pandas as pd
import matplotlib.pyplot as plt

from numpy import argmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.utils import plot_model
# from imblearn.over_sampling import RandomOverSampler as ros


heat_map_dir = 'result/attention.png'  # 热图路径

n_inputs = embedding_size  # 输入维度，等于embedding大小
n_steps = time_step

n_classes = 2

# input_keep_prob = 0.6  # dropout层
CONTINUE_TRAIN = True

# 超参数
lr = 0.01
# 200000  # 100000  # 迭代次数（不是epoch数） epoch=training_iters/data_len
training_iters = 1000000


random_state = 0
tf.random.set_seed(0)


class PositionEmbedding(layers.Layer):
    '''
    attention is all you need 论文中的 position embedding层
    继承keras layer
    h2 = PositionEmbedding(time_step, embedding_size)(h1)
    '''

    def __init__(self, position, d_model, name="PositionEmbedding", **kwargs):
        super().__init__(name=name, **kwargs)

        # 储存用于恢复模型的init参数，用在get_config函数中
        #####
        self.position = position
        self.d_model = d_model
        #####

        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000,
                            (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):

        config = super().get_config().copy()
        config.update({'position': self.position, 'd_model': self.d_model})
        return config


class Add_weight(layers.Layer):
    '''
    keras层
    为input添加一个可训练权重 => input · weight
    '''

    def __init__(self, name="Add_weight", **kwargs):  # 申请、储存本层需要用到的属性、对象等
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):  # 需要根据input_shape改变配置时要重写的函数
        # 添加的可训练的权重

        self.weight_to_mut = tf.Variable(tf.constant(
            0.1, shape=[input_shape[2], input_shape[2]]),
            trainable=True)
        super().build(input_shape)  # 一定要在最后调用它

    def call(self, x):  # 调用该层时进行的运算
        return K.dot(x, self.weight_to_mut)  # 点乘

    def get_config(self):  # 返回初始化变量，用于模型读取时使用
        config = super().get_config().copy()
        return config

    def compute_output_shape(self, input_shape):  # 返回的矩阵大小
        return (input_shape[2], input_shape[2])


def build_model(lr=0.001):
    # inputs = tf.keras.Input(shape=(
    #     n_steps,
    #     n_inputs,
    # ))  # 不要指定batch_size大小！！！

    inputs_all = tf.keras.Input(shape=(
        n_steps*n_inputs+low_feature_dim
    ))

    inputs = tf.keras.layers.Lambda(
        lambda x: x[:, :-low_feature_dim])(inputs_all)

    inputs = layers.Reshape((n_steps, n_inputs))(inputs)

    inputs_low = tf.keras.layers.Lambda(
        lambda x: x[:, -low_feature_dim:])(inputs_all)

    # weights

    input_wq = Add_weight(name='add_1')(inputs)
    input_wv = Add_weight(name='add_2')(inputs)
    input_wk = Add_weight(name='add_3')(inputs)

    h1 = layers.Attention()([input_wq, input_wv,
                             input_wk])  # self-attention [query,value,key]

    h2 = PositionEmbedding(time_step, embedding_size,
                           name="PositionEmbedding")(h1)  # 自定义层设置name

    h3 = layers.Flatten()(h2)  # 展开后使用全连接

    h3_con = layers.Concatenate()([h3, inputs_low])  # 拼接低维特征

    h4 = layers.Dense(n_classes, input_shape=(
        time_step * embedding_size+low_feature_dim,))(h3_con)

    outputs = layers.Activation('softmax')(h4)

    new_model = tf.keras.Model(inputs=inputs_all, outputs=outputs)

    new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
    return new_model


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


def train_att(
        vx_train,
        vx_test,
        vy_train,
        vy_test,
        model_save_dir,
        CONTINUE_TRAIN=CONTINUE_TRAIN,
        lr=lr,
        training_iters=training_iters,
        BATCH_SIZE=BATCH_SIZE,  # 128 训练时批的大小
        write_result_to_file=False,
        result_file='result/imbalance.txt',
        IS_GAN=False,
        GAN_DIR='model/attention_weights.keras'
    # VAL_SIZE=VAL_SIZE,
):
    '''
    训练attention神经网络
    '''
    train_model = build_model(lr=lr)

    if IS_GAN:
        train_model.load_weights(GAN_DIR)
    elif CONTINUE_TRAIN:
        # train_model = tf.keras.models.load_model('model/attention.keras', custom_objects={
        #     'PositionEmbedding': PositionEmbedding,
        #     'add_1':Add_weight,
        #     'add_2':Add_weight,
        #     'add_3':Add_weight,
        # })
        train_model.load_weights(model_save_dir)

    # train_model.summary()
    # plot_model(train_model, to_file='result/attention.png',
    #            show_shapes=True, show_layer_names=True)
    # exit()

    # label编码器
    values = np.array([0, 1])
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(values)

    x_train, x_test, y_train, y_test = vx_train, vx_test, vy_train, vy_test

    # 重采样
    # ros = ros(random_state=0)
    # x_train, y_train = ros.fit_resample(
    #     x_train.reshape(-1, time_step*embedding_size), y_train)

    x_train = x_train.reshape(-1, time_step * embedding_size+low_feature_dim)

    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    # train_model.fit(x_train, y_train, batch_size=BATCH_SIZE,
    #           epochs=training_iters//x_train.shape[0], validation_split=0.1, callbacks=[early_stopping], shuffle=True)

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

    # 由于TF2的尿性，不能直接使用val分数，需要使数据在经过分割后训练集恰好能被batch整除，否则报错；很烦，这里取消earlystop
    # train_model.fit(x_train, y_train, batch_size=BATCH_SIZE,
    #           epochs=training_iters//x_train.shape[0], validation_split=0.2, callbacks=[early_stopping], shuffle=True)
    # # train_model.save(model_save_dir)
    train_model.save_weights(model_save_dir, overwrite=True)

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

    print()
    print('Test acc:', accuracy_score(y_test_real, y_pred_real))
    print('Test recall:', recall_score(y_test_real, y_pred_real))
    print('Test precision:', precision_score(y_test_real, y_pred_real))
    print('Test f1-score:', f1_score(y_test_real, y_pred_real))

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

    # conf_max(y_test_real, y_pred_real)


def eval_result(x_test,
                y_test,
                model_save_dir='model/attention_weights.keras',
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

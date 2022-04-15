# -*- coding: utf-8 -*-

from discriminator.data_proc import *
from discriminator.cnn_tf2 import *
from discriminator.word2vec import *

import shutil

import collections
import gc

from vectorize import vectorize_ob_dataset_from_dir


threadnum = 200  # 混淆webshell时的多线程个数；500超出内存大小（300也超？


if __name__ == '__main__':

    # 清空记录
    fw = open('result/gan_train_cnn_before.txt', 'w')  # 清空记录文件
    fw.close()

    fw = open('result/gan_obfuscated_cnn_eval.txt', 'w')  # 清空记录文件
    fw.close()

    fw = open('result/gan_train_cnn_eval.txt', 'w')  # 清空记录文件
    fw.close()

    for i in range(gan_epoch):
        fw = open('result/gan_train_cnn_' + str(i) + '.txt', 'w')  # 清空记录文件
        fw.close()

    fw = open('result/ob_shell_cnn_num.txt', 'w')  # 清空记录文件
    fw.close()

    try:
        shutil.rmtree("data/obfuscated_bak_cnn/")
    except FileNotFoundError:
        pass
    os.mkdir("data/obfuscated_bak_cnn/")  # 清空文件夹

    try:
        shutil.rmtree("tmp")
    except FileNotFoundError:
        pass
    os.mkdir("tmp")  # 清空文件夹

    try:
        shutil.rmtree("result/ob_actions_cnn/")
    except FileNotFoundError:
        pass
    os.mkdir("result/ob_actions_cnn/")  # 清空文件夹
    ####################

    vx_train = np.load(x_train_dir)
    vx_test = np.load(x_test_dir)
    vy_train = np.load(y_train_dir)
    vy_test = np.load(y_test_dir)

    test_counter = collections.Counter(vy_test)
    train_counter = collections.Counter(vy_train)

    with open('result/orig_data_cnn_num.txt', 'w') as fw:  # 记录原始数据的数量
        print('test counter: ', test_counter, file=fw)
        print('train counter: ', train_counter, file=fw)

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

    train_cnn(
        x_train,
        x_test,
        y_train,
        y_test,
        model_save_dir='model/cnn_weights.keras',
        CONTINUE_TRAIN=False,
        lr=0.001,
        BATCH_SIZE=1024,
        training_iters=1000000,
        write_result_to_file=True,
        result_file='result/gan_train_cnn_before.txt')  # 使用初始数据训练数据集（之后需要改为训练D 3次）

    # 训练完成后才能调用检测函数
    from discriminator.detect_cnn_tf2 import *

    # 训练完模型后才能引入，否则引入时env会调用训练的模型，而此时模型还未训练好
    from shellgenerator.ga_bypass_cnn_train import *

    print('GAN epoch', 0)

    model.load_weights('model/cnn_weights.keras')  # 更新ENV中的D的网络参数

    ga_bypass_cnn_train(ga_epoch,
                        'bin/ob_ga_cnn_code.pkl',
                        max_encode_time=max_encode_time)  # 需要先训练G才能import bypass_cnn_test

    from shellgenerator.ga_bypass_cnn_test import *
    from shellgenerator.orig_shells import orig_shell_li


    # print('用于混淆的原webshell数：', len(orig_shell_li))
    print('The num of original webshells to be obfuscated: ', obshell_num)
    random.shuffle(orig_shell_li) # 随机选择obshell_num个原始样本用于混淆
    orig_shell_li_ob=orig_shell_li[:obshell_num]

    print(1)
    for epoch in range(gan_epoch):
        try:
            shutil.rmtree("data/obfuscated_cnn")
        except FileNotFoundError:
            pass
        os.mkdir("data/obfuscated_cnn")  # 清空文件夹

        
        for vori_shell in orig_shell_li_ob:

            temp_file = 'tmp/' + hex(random.randint(
                0, 1000000000000))[2:] + '.php'
            with open(temp_file, 'w') as fw:
                fw.write(vori_shell)
            temp_ops = load_php_opcode(temp_file)
            os.remove(temp_file)

            if temp_ops == '':
                continue
            
            ga_bypass_cnn_test(
                vori_shell, save_dir='bin/ob_ga_cnn_code.pkl')
        

        actions_log('result/ob_actions_cnn/gan_' + str(epoch) +
                    '.txt', action_len=max_encode_time, shell_root="data/obfuscated_cnn")  # 记录混淆所用的函数

        print(2)

        with open('result/ob_shell_cnn_num.txt', 'a') as fw:
            print('gan epoch:', epoch, file=fw)

        # 向量化混淆的webshell为numpy

        vectorize_ob_dataset_from_dir('data/obfuscated_bak_cnn/', 'bin/x_cnn_obfuscated.npy',
                                      'bin/y_cnn_obfuscated.npy',)

        x_obfuscated = np.load('bin/x_cnn_obfuscated.npy')
        y_obfuscated = np.load('bin/y_cnn_obfuscated.npy')

        with open('result/ob_shell_cnn_num.txt', 'a') as fw:
            print('[num] 本轮混淆shell数据量为：', len(y_obfuscated), file=fw)

        y_obfuscated = onehot_encoder.transform(y_obfuscated.reshape(
            -1, 1))  # onehot编码

        # 混淆样本少于100时，将混淆样本扩大n倍；if the ob shells is too little, repeat them n times to emphasis them
        if y_obfuscated.shape[0]<100:
            x_obfuscated = np.repeat(x_obfuscated, too_little_repeat_time, axis=0)
            y_obfuscated = np.repeat(y_obfuscated, too_little_repeat_time, axis=0)

        # 混淆训练时，需要拼接一部分白样本，否则训练会偏向只输出黑样本结果（极端的不平衡问题）

        t_x_train = np.append(x_train, x_obfuscated)  # 将混淆的数据添加至训练数据
        t_y_train = np.append(y_train, y_obfuscated).reshape(-1, 2)

        with open('result/gan_obfuscated_cnn_eval.txt', 'a') as fw:
            print('GAN epoch: ' + str(epoch), file=fw)

        gan_dir = 'model/cnn_weights.keras' if epoch == 0 else 'model/cnn_weights_' + str(
            epoch - 1) + '.keras'  # 对上一次训练的模型进行训练

        eval_result(
            x_obfuscated,
            y_obfuscated,
            model_save_dir=gan_dir,
            if_conf_max=False,
            write_to_file=True,
            result_file='result/gan_obfuscated_cnn_eval.txt')  # 评估混淆的webshell的召回率

        # 添加了混淆数据后再次训练D
        train_cnn(
            t_x_train,
            x_test,
            t_y_train,  # append后需要重新调整维度
            y_test,
            'model/cnn_weights_' + str(epoch) + '.keras',  # 模型储存位置
            lr=0.0001,
            CONTINUE_TRAIN=True,
            IS_GAN=True,
            GAN_DIR=gan_dir,
            write_result_to_file=True,
            result_file='result/gan_train_cnn_' + str(epoch) + '.txt',
            training_iters=300000)

        # 评估并记录重新训练后的召回率
        with open('result/gan_train_cnn_eval.txt', 'a') as fw:
            print('GAN epoch: ' + str(epoch), file=fw)

        eval_result(x_obfuscated,
                    y_obfuscated,
                    model_save_dir='model/cnn_weights_' + str(epoch) +
                    '.keras',
                    write_to_file=True,
                    result_file='result/gan_train_cnn_eval.txt')  # 评估重新训练后的召回率

        print('GAN epoch', epoch + 1)

        model.load_weights('model/cnn_weights_' + str(epoch) +
                           '.keras')  # 更新ENV中的D的网络参数

        ga_bypass_cnn_train(
            ga_epoch, 'bin/ob_ga_cnn_code.pkl', max_encode_time=max_encode_time)  # 训练G 1次，小更新率 {{要更新ENV中的网络参数}}
        gc.collect()

# from shellgenerator.ga_bypass_dnn_train import ga_bypass_dnn_train
import pickle
# import numpy

from .Shell_encoder import Shell_encoder
from discriminator.detect_dnn_tf2 import *
from func_timeout import FunctionTimedOut

# ga_bypass_dnn_train(1,'bin/ob_ga_code.pkl')

# with open('bin/ob_ga_code.pkl','rb') as fr:
#     ga_code_li=pickle.load(fr)

# print(ga_code_li)


def ga_bypass_dnn_test(orig_shell, save_dir='bin/ob_ga_code.pkl'):
    '''
    利用寻找到的ga编码混淆orig_shell

    注意要先加载最新的检测model
    '''

    web_encoder = Shell_encoder(orig_shell)

    with open(save_dir, 'rb') as fr:
        ga_code_li = pickle.load(fr)

    for ga_code in ga_code_li:
        try:
            ob_shell = web_encoder.code_shell_encode(ga_code)
        except FunctionTimedOut:
            ob_shell = '<?php ?'

        temp_shell = ob_shell
        temp_shell_vec = tovector(temp_shell)
        if temp_shell_vec is None: # the ob shell is crashed
            continue
        good_rate = dnn_logit(
            temp_shell_vec.reshape(
                1, time_step* embedding_size+low_feature_dim))[0][0]  # 初始化良性值为]原始webshell的良性值

        if good_rate > 0.5:  # 如果良性率大于0.5，则混淆成功，保存好文件
            # {{}}写入文件后，新的一轮需要清空混淆数据（GAN每一轮generator都产生新数据

            actions_str = [str(int(vstr)) for vstr in ga_code]
            temp_file = '_'.join(actions_str) + '_' + hex(
                random.randint(0, 1000000000000))[2:] + '.php'
            with open('data/obfuscated_dnn/' + temp_file, 'w') as fw:
                fw.write(ob_shell)
            with open('data/obfuscated_bak_dnn/' + temp_file, 'w') as fw:
                fw.write(ob_shell)

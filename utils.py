# -*- coding: utf-8 -*-

import os
import random
import string
import re

import subprocess
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

from threading import Lock

USE_GPU = True

rand_seed = 1

BATCH_SIZE = 512  # 1024  # 训练时批的大小

# 存放op操作符字符串的位置
good_ops_dir_unwash = 'data/good_op_unwash.csv'
bad_ops_dir_unwash = 'data/bad_op_unwash.csv'

good_ops_dir = 'data/good_op.csv'
bad_ops_dir = 'data/bad_op.csv'

# benign dataset
# phpgoodroot = 'data/good_test'
phpgoodroot = 'data/good_collect_all'

# web shell dataset
# phproot = 'data/bad_test'
phproot = 'data/bad_collect_all'

# web shells for ga
# phproot_ga='data/bad_test'
phproot_ga = 'data/bad_ga'

# -- [!!!!!!!!!!!] 参数区 the parameters you should set before running the code

ENCODE_TIMEOUT = 60 * 60  # 1*60  # stop timeout task by seconds

# the num of the benign samples in all of the dataset(to reduce class imbalance, we do not use all the benigh samples)
benign_num = 4379  # 10000

gan_epoch = 10  # 8 # the epoch of GAN

ga_epoch = 2  # the epoch of shell obfuscate; 一轮训练GA得到的混淆方法组合的个数

# if the ob shells is too little, repeat them to emphasis them
too_little_repeat_time = 20

# the num of the origin shells that are used to be obfuscated in every epoch
obshell_num = 1000

max_encode_time = 12  # 10 编码次数控制在20次以内

NIND = 100  # 30 # individual num
MAX_GEN = 20  # 20 # evolution generation

embedding_size = 100  # 隐层的维度
vec_dir = "bin/word2vec.model"  # word2vec存放位置
window = 5  # 上下文距离; word2vec window
iter_num = 30  # word2vec的迭代数;the iteration times of word2vec
min_num = 1  # word2vec的最少出现次数
max_voc = 1000  # 最大字典数
time_step = 200  # 时序，即单句的最大seg长

threadnum = 10  # 混淆webshell时的多线程个数

low_feature_dim = 4  # 低维特征的维度; the dimension of the statistical features

lens_dir = 'bin/lens.pkl'  # 存放所有payload的长度统计
y_train_dir = 'bin/y_train.npy'
x_train_dir = 'bin/x_train.npy'

y_test_dir = 'bin/y_test.npy'
x_test_dir = 'bin/x_test.npy'

# -- [!!!!!!!!!!!] 参数区结束;parameters field end

modelLock = Lock()  # 访问模型的锁

random.seed(rand_seed)


if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 正则

# 匹配需要eval或需要其他执行的内容
eval_thing_par = re.compile(
    r'(?:eval|shell_exec|exec|system|passthru|popen|proc_open|assert)\(([^\)]+)\)')

# 匹配整个eval语句
eval_thing_ic_par = re.compile(
    r'(?:eval|shell_exec|exec|system|passthru|popen|proc_open|assert)\([^\)]+\)')

eval_thing2_par = re.compile(r'`([^`]+)`')  # 匹配``之间要执行的变量

# 匹配整个eval语句
eval_thing2_ic_par = re.compile(r'([^\s^;^{^}]*`[^`]+`[^;]*)')

# 匹配所有变量名：['a', 'b', '_POST', 'b', 'a']
variable_par = re.compile(r'\$([\w|\_]+)[=|\)|`|;|\s|\)|\[|{|}|\'|"]')

# 匹配所有原生全局变量（GET、POST等） ：['$_POST{123}']
magic_par = re.compile(r'(\$_[\w|\_|\[|\]|{|}|\'|"]+)[=|\)|`|;|\s|\)|\[]')

# 匹配敏感函数
# eval_func_par = re.compile(
#     r'\s(?:shell_exec|exec|system|passthru|popen|proc_open|assert)')

eval_func_par = re.compile(
    r'[\s;}](?:shell_exec|exec|system|passthru|popen|proc_open|assert)\(')
# eval在php7中不属于函数，不能使用异或调用，如果使用php7需要去掉eval

# 匹配opcode

op_par = re.compile(r'\s(\b[A-Z_]+\b)\s')


# 匹配if
if_par = re.compile(r'[\s;}]if\s*\(')

# for
for_par = re.compile(r'[\s;}]for\s*\(')
# while
while_par = re.compile(r'[\s;}]while\s*\(')


def get_eval_fun_pos(orig_shell):
    '''
    Get the eval function 

    获取执行函数
    '''
    try:
        eval_func = re.search(eval_func_par, orig_shell).span()
        eval_func = list(eval_func)
        eval_func[1] -= 1  # 去掉'('

        while not orig_shell[eval_func[0]].isalpha():  # 去掉非字母的符号

            eval_func[0] += 1

    except Exception:
        eval_func = None

    return eval_func


def get_eval_thing(orig_shell):
    '''
    Get the content to be evaled .

    获取将执行的内容
    '''
    try:
        if '`' in orig_shell:
            eval_thing = re.findall(eval_thing2_par, orig_shell)[0]
        else:
            eval_thing = re.findall(eval_thing_par, orig_shell)[0]
    except IndexError:
        # print('''[w] Couldn't find eval thing, return \'Not exist string\'''')
        return 'Not exist string'

    return eval_thing


def get_eval_thing_ic(orig_shell):
    '''
    Get the eval code.

    获取整个命令执行内容
    '''
    try:
        if '`' in orig_shell:
            eval_thing = re.findall(eval_thing2_ic_par, orig_shell)[0]
        else:
            eval_thing = re.findall(eval_thing_ic_par, orig_shell)[0]
    except IndexError:
        # print('''[w] Couldn't find eval thing, return \'Not exist string\'''')
        return 'Not exist string'

    return eval_thing.replace('<?php ', '')


def random_string(stringLength=10):
    '''
    get a random string
    随机产生一个字母组成的字符串
    '''
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def random_keys(len):
    '''
    get a random key
    随机产生一个长度为len的字符串
    '''
    str = '`~-=!@#$%^&*_/+?<>{}|:[]abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.sample(str, len))


def load_php_opcode(phpfilename, remove_li=[]):
    """
    Return the php opcode.

    获取php opcode 信息；提取一个php文件为opcode操作符连接的句子
    """

    try:
        output = str(
            subprocess.check_output(
                ['php', '-dvld.active=1', '-dvld.execute=0', '-dvld.verbosity=0',
                 '-dvld.skip_prepend=1', '-dvld.skip_append=1', phpfilename],
                stderr=subprocess.STDOUT))

        tokens = re.findall(r'\s(\b[A-Z_]+\b)\s', output)  # opcode操作符提取正则

        t = tokens[0]
        for tok in tokens[1:]:
            if tok not in remove_li:
                t = t+' '+tok
            # else:
            #     print(tok)
            #     exit()
                # pass
        # t = " ".join(tokens)
        return t.replace('E O E ', '')  # 由于opcode正则会匹配每个func开头的非opcode字符，在这里去除
    except Exception as e:
        # print('[Error] ', phpfilename, ' Error: ', e)
        # exit()
        return ""  # 未读取成功或没有任何操作符时


def shuffle_data(train_data, train_target, random_seed=0):
    '''
    打乱数据和标签对
    '''
    batch_size = len(train_target)
    index = [i for i in range(0, batch_size)]
    np.random.seed(random_seed)  # 固定随机种子
    np.random.shuffle(index)
    batch_data = []
    batch_target = []
    for i in range(0, batch_size):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data, batch_target


def is_in_fields(num, fields):
    '''
    传入：数字，例：1，范围列表，例：[[0,1],[5,7]]
    返回数字是否在给出的范围内，若在，则返回在第几个范围
    '''
    i = 0
    for field in fields:
        if num > field[0] and num < field[1]:
            return i
        i += 1
    return False


def get_one_class_field(code_seg):
    '''
    寻找下一个class的起止
    首先找到一个 "class" 再寻找第一个'{'对应的'}'（需要消减中间多的括号）

    返回相对起始位置（'class '开始），相对结束位置（由于是截取的代码）
    '''

    class_pos = code_seg.find('class ')
    if class_pos == -1:  # 不存在class语句
        return False, False

    first_pos = code_seg.find('{', class_pos + 5)

    i = first_pos + 1
    flag = 1

    comp_time = 0
    while flag != 0 and comp_time < 1000000:

        if code_seg[i] == '{':
            flag += 1
        elif code_seg[i] == '}':
            flag -= 1
        i += 1
        comp_time += 1
    class_end = i - 1

    return class_pos, class_end


def get_class_fields(phpcode):
    '''
    统计出代码中 class的所有范围
    '''
    class_fields = []
    code_seg = phpcode

    index_now = 0

    find_time = 0
    while True:
        class_pos, class_end = get_one_class_field(code_seg)
        if class_pos is False:
            return class_fields

        class_fields.append([class_pos + index_now, class_end + index_now])

        index_now += class_end
        code_seg = code_seg[class_end:]
        find_time += 1
        if find_time > 1000:  # 超过寻找次数，退出
            return


def get_one_if_field(code_seg):
    '''
    寻找下一个if的起止
    首先找到一个 "if" 再寻找第一个'{'对应的'}'（需要消减中间多的括号）

    返回相对起始位置（'if'开始），相对结束位置（由于是截取的代码）
    '''

    try:
        if_pos = re.search(if_par, code_seg).span()[0]

    except Exception:  # 不存在if语句
        return False, False

    first_pos = code_seg.find('(', if_pos + 1)

    i = first_pos + 1
    flag = 1

    comp_time = 0
    while flag != 0 and comp_time < 1000000:

        if code_seg[i] == '(':
            flag += 1
        elif code_seg[i] == ')':
            flag -= 1
        i += 1
        comp_time += 1
    if_end = i - 1

    return if_pos, if_end


def get_if_fields(phpcode):
    '''
    统计出代码中 if的所有范围
    '''
    if_fields = []
    code_seg = phpcode

    index_now = 0

    find_time = 0
    while True:
        if_pos, if_end = get_one_if_field(code_seg)
        if if_pos is False:
            return if_fields

        if_fields.append([if_pos + index_now, if_end + index_now])

        index_now += if_end
        code_seg = code_seg[if_end:]
        find_time += 1
        if find_time > 1000:  # 超过寻找次数，退出
            return


def get_one_for_field(code_seg):
    '''
    寻找下一个for的起止
    首先找到一个 "for" 再寻找第一个'{'对应的'}'（需要消减中间多的括号）

    返回相对起始位置（'for'开始），相对结束位置（由于是截取的代码）
    '''

    try:
        for_pos = re.search(for_par, code_seg).span()[0]

    except Exception:  # 不存在for语句
        return False, False

    # for_pos = code_seg.find('for')
    # if for_pos == -1:  # 不存在for语句
    #     return False, False

    first_pos = code_seg.find('(', for_pos + 1)

    i = first_pos + 1
    flag = 1

    comp_time = 0
    while flag != 0 and comp_time < 1000000:

        if code_seg[i] == '(':
            flag += 1
        elif code_seg[i] == ')':
            flag -= 1
        i += 1
        comp_time += 1
    for_end = i - 1

    return for_pos, for_end


def get_for_fields(phpcode):
    '''
    统计出代码中 for的所有范围
    '''
    for_fields = []
    code_seg = phpcode

    index_now = 0

    find_time = 0
    while True:
        for_pos, for_end = get_one_for_field(code_seg)
        if for_pos is False:
            return for_fields

        for_fields.append([for_pos + index_now, for_end + index_now])

        index_now += for_end
        code_seg = code_seg[for_end:]
        find_time += 1
        if find_time > 1000:  # 超过寻找次数，退出
            return


def get_one_while_field(code_seg):
    '''
    寻找下一个while的起止
    首先找到一个 "while" 再寻找第一个'{'对应的'}'（需要消减中间多的括号）

    返回相对起始位置（'while'开始），相对结束位置（由于是截取的代码）
    '''

    try:
        while_pos = re.search(while_par, code_seg).span()[0]

    except Exception:  # 不存在while语句
        return False, False

    # while_pos = code_seg.find('while')
    # if while_pos == -1:  # 不存在while语句
    #     return False, False

    first_pos = code_seg.find('(', while_pos + 1)

    i = first_pos + 1
    flag = 1

    comp_time = 0
    while flag != 0 and comp_time < 1000000:

        if code_seg[i] == '(':
            flag += 1
        elif code_seg[i] == ')':
            flag -= 1
        i += 1
        comp_time += 1
    while_end = i - 1

    return while_pos, while_end


def get_while_fields(phpcode):
    '''
    统计出代码中 while的所有范围
    '''
    while_fields = []
    code_seg = phpcode

    index_now = 0

    find_time = 0
    while True:
        while_pos, while_end = get_one_while_field(code_seg)
        if while_pos is False:
            return while_fields

        while_fields.append([while_pos + index_now, while_end + index_now])

        index_now += while_end
        code_seg = code_seg[while_end:]
        find_time += 1
        if find_time > 1000:  # 超过寻找次数，退出
            return


def xor(c1, c2):
    return hex(ord(c1) ^ ord(c2)).replace('0x', r"\x")


def bit_not(c1):
    return hex(~ord(c1) + 0xff + 1).replace('0x', r"\x")


def actions_log(wr_dir, action_len=5, shell_root='data/obfuscated/'):
    '''
    log the obfuscation methods used in the gan
    统计obfuscated目录下的混淆shell所用的混淆方法actions
    '''
    actions_li = []
    for root, dirs, files in os.walk(shell_root):
        for vfile in files:
            actions = vfile.split('_')[:action_len]
            actions_li.append(str(actions))
    actions_li = list(set(actions_li))

    with open(wr_dir, 'w') as fw:
        for action in actions_li:
            fw.write(action + '\n')


def sep_data(good_dir, bad_dir, test_ratio=0.2):
    '''
    split the data set into train and test sets by the test_ratio.
    根据测试集比例分割数据集，并记录训练集和测试集至 data_sep/
    '''
    print('[+ message] Loading good data from {} and bad data from {}'.format(good_dir, bad_dir))

    good_li = []
    bad_li = []

    for root, dirs, files in os.walk(good_dir):
        for filename in files:
            good_li.append(os.path.join(root, filename))

    for root, dirs, files in os.walk(bad_dir):
        for filename in files:
            bad_li.append(os.path.join(root, filename))

    # shuffle the data to seperate the train and test samples
    random.shuffle(good_li)
    random.shuffle(bad_li)

    good_li = good_li[:benign_num]

    good_train_num = int(len(good_li)*(1-test_ratio))
    bad_train_num = int(len(bad_li)*(1-test_ratio))
    # print(good_train_num)

    good_train_li = good_li[:good_train_num]
    good_test_li = good_li[good_train_num:]

    bad_train_li = bad_li[:bad_train_num]
    bad_test_li = bad_li[bad_train_num:]

    with open('data/data_sep/train/train.txt', 'w') as fw:
        for sample in good_train_li:
            fw.write(sample+',0\n')
        for sample in bad_train_li:
            fw.write(sample+',1\n')

    with open('data/data_sep/test/test.txt', 'w') as fw:
        for sample in good_test_li:
            fw.write(sample+',0\n')
        for sample in bad_test_li:
            fw.write(sample+',1\n')


def trans_php_file_opcode(write_dir='data\\train_op.csv'):
    '''
    遍历训练集'data/data_sep/train/train.txt'中所有的PHP文件，
    将其转化为操作符序列并写入到文件中
    '''
    # print('开始生成 {} 路径中的PHP的opcode操作码文件'.format(dir))

    with open(write_dir, 'w') as fw:

        files = open('data/data_sep/train/train.txt').readlines()

        for vfile in files:
            filename = vfile.split(',')[0]
            try:

                file_content = load_php_opcode(filename, ['EXT_STMT'])
                if (file_content == ''):  # 空文件或读取失败时跳过该文件
                    print(
                        f'[+ error] File {filename} cannot be execed by php.')
                    continue
                fw.write(file_content + '\n')
            except:
                continue


if __name__ == '__main__':
    pass

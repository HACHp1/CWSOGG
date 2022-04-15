from CFG.CFG import get_cfgs_fromfile
from urllib.parse import unquote
from gensim.models.word2vec import Word2Vec
from utils import *

# model=Word2Vec.load(vec_dir)


def get_stat_feat(phpfile):
    '''
    input:phpfile directory
    output: statistical features

    * 超级全局变量的数量。在webshell中，计算超全局变量的数量，包括: `$_GET, $_POST, $_FILES, $_COOKIE, $_REQUEST, $_SERVER, $_SESSION`。
    * 跳转语句的百分比。
    * 字符串数量。
    * 最长的字符串的长度。
    '''
    jmp_op = [b'JMP', b'JMPZ', b'JMPNZ', b'JMPZNZ', b'JMPZ_EX', b'JMPNZ_EX']

    cfgs, _ = get_cfgs_fromfile(phpfile)

    global_va_num = 0
    jmp_num = 0
    str_num = 0
    max_str_len = 0

    instru_num = 0

    for bbs in cfgs:
        for bb in bbs:
            for instru in bb:
                instru_num += 1

                if_str = False

                if instru[2] == b'global':
                    global_va_num += 1
                if instru[1] in jmp_op:
                    jmp_num += 1

                elif instru[1] == b'ASSIGN' and instru[5][-1] == b'\'':
                    if_str = True
                    str_num += 1
                    tmp_str = instru[5][instru[5].find(b'\'')+1:-1]
                elif instru[1] == b'ECHO' and instru[5][-1] == b'\'':
                    if_str = True
                    str_num += 1
                    tmp_str = instru[5][instru[5].find(b'\'')+1:-1]
                elif instru[1] == b'CONCAT' and instru[5][-1] == b'\'':
                    if_str = True
                    str_num += 1
                    tmp_str = instru[5][instru[5].find(b'\'')+1:-1]

                if if_str:
                    tmp_str = unquote(tmp_str)
                    # print(tmp_str)
                    tmp_len = len(tmp_str)
                    if tmp_len > max_str_len:
                        max_str_len = tmp_len
    if instru_num == 0:
        jmp_ratio = 0
    else:
        jmp_ratio = jmp_num/instru_num

    return [global_va_num, jmp_ratio, str_num, max_str_len]


def vectorize_php_dataset(data_dir='data/data_sep/train/train.txt', save_x_dir='bin/x_train.npy', save_y_dir='bin/y_train.npy'):
    '''
    将数据集全部转化为向量，并保存
    '''
    with open(data_dir) as fr:
        php_dirs = fr.readlines()

    x = []
    y = []
    tempvx = []
    # payloads_seged = []
    model = Word2Vec.load(vec_dir)

    for phpdir_and_label in php_dirs:
        [filedir, label] = phpdir_and_label.split(',')
        payload = load_php_opcode(filedir)

        if payload == '':
            print(f'[+error] file {filedir} cannot be execed by php')
            continue

        tempseg = payload.split(' ')
        if (tempseg == []):
            print(payload[:100])

        for word in tempseg:
            try:
                tempvx.append(model.wv.get_vector(word))
            except KeyError as e:
                tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位

        tempvx = np.array(tempvx)
        if (tempvx.shape[0] == 0):
            print(payload[:100])

        # 字符串向量长度填充
        if (tempvx.shape[0] < time_step):
            try:
                tempvx = np.pad(tempvx, ((0, time_step - tempvx.shape[0]), (0, 0)),
                                'constant',
                                constant_values=0)
            except ValueError as e:
                print(filedir)
                print(tempvx.shape)
                print(tempvx)
                exit()
        elif (tempvx.shape[0] > time_step):
            tempvx = tempvx[0:time_step]

        low_fea = np.array(get_stat_feat(filedir))

        tempvx = np.concatenate((tempvx.flatten(), low_fea))
        x.append(tempvx)
        y.append(int(label.strip()))

        tempvx = []

    print('[+num] 数据量为：', len(y))

    x = np.array(x)
    y = np.array(y)

    np.save(save_x_dir, x)
    np.save(save_y_dir, y)

    # with open(save_x_dir, 'wb') as f:
    #     pickle.dump(x, f)

    # with open(save_y_dir, 'wb') as f:
    #     pickle.dump(y, f)

    return


def vectorize_ob_dataset_from_dir(payloads_dir, save_x_dir='bin/x_train.npy', save_y_dir='bin/y_train.npy'):
    '''
    将混淆目录下的数据集全部转化为向量，并保存
    '''

    php_dirs = []
    for root, dirs, files in os.walk(payloads_dir):
        for filename in files:
            php_dirs.append(os.path.join(root, filename))

    x = []
    y = []
    tempvx = []
    # payloads_seged = []
    model = Word2Vec.load(vec_dir)

    for filedir in php_dirs:

        payload = load_php_opcode(filedir)

        if payload == '':
            print(f'[+error] file {filedir} cannot be execed by php')
            continue

        tempseg = payload.split(' ')
        if (tempseg == []):
            print(payload[:100])

        for word in tempseg:
            try:
                tempvx.append(model.wv.get_vector(word))
            except KeyError as e:
                tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位

        tempvx = np.array(tempvx)
        if (tempvx.shape[0] == 0):
            print(payload[:100])

        # 字符串向量长度填充
        if (tempvx.shape[0] < time_step):
            try:
                tempvx = np.pad(tempvx, ((0, time_step - tempvx.shape[0]), (0, 0)),
                                'constant',
                                constant_values=0)
            except ValueError as e:
                print(filedir)
                print(tempvx.shape)
                print(tempvx)
                exit()
        elif (tempvx.shape[0] > time_step):
            tempvx = tempvx[0:time_step]

        low_fea = np.array(get_stat_feat(filedir))

        tempvx = np.concatenate((tempvx.flatten(), low_fea))
        x.append(tempvx)
        y.append(1)

        tempvx = []

    print('[+num] 数据量为：', len(y))

    x = np.array(x)
    y = np.array(y)

    np.save(save_x_dir, x)
    np.save(save_y_dir, y)
    return


def php_one_vectorize(phpfile, w2v_model):
    '''
    input:php file dir,w2v model
    output:the vector of the file
    '''
    tempvx = []
    payload = load_php_opcode(phpfile)

    if payload == '':
        print(f'[+ error] file cannot be execed by php: {phpfile}')
        return
        raise Exception(f'[+ error] file cannot be execed by php: {phpfile}')

    tempseg = payload.split(' ')
    if (tempseg == []):
        print(payload[:100])

    for word in tempseg:
        try:
            tempvx.append(w2v_model.wv.get_vector(word))
        except KeyError as e:
            tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位

    tempvx = np.array(tempvx)
    if (tempvx.shape[0] == 0):
        print(payload[:100])

    # 字符串向量长度填充
    if (tempvx.shape[0] < time_step):
        try:
            tempvx = np.pad(tempvx, ((0, time_step - tempvx.shape[0]), (0, 0)),
                            'constant',
                            constant_values=0)
        except ValueError as e:
            print(phpfile)
            print(tempvx.shape)
            print(tempvx)
            exit()
    elif (tempvx.shape[0] > time_step):
        tempvx = tempvx[0:time_step]

    low_fea = np.array(get_stat_feat(phpfile))
    tempvx = np.concatenate((tempvx.flatten(), low_fea))

    return tempvx


if __name__ == '__main__':

    'data/data_sep/train/train.txt'
    'data/data_sep/test/test.txt'
    # sep_data('data/good_test', 'data/bad_test', 0.2) # 分割训练集测试集

    # from discriminator.dnn_tf2 import build_model

    # model = build_model()
    # model.summary()

    # from tensorflow.keras.utils import plot_model
    # plot_model(model, to_file='result/dnn.png',
    #            show_shapes=True, show_layer_names=True)

    model = Word2Vec.load(vec_dir)
    x = php_one_vectorize(
        'data/bad_test/0a4d17d0b3e6551cde71a09cc2070490.php', model)
    exit()
    vectorize_php_dataset('data/data_sep/train/train.txt',
                          x_train_dir, y_train_dir)

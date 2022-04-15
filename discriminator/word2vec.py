'''
功能：
1. 使用word2vec将字符串文件向量化
2. 将向量化的结果储存在bin路径下
'''

from gensim.models.word2vec import Word2Vec
from utils import *
import pickle
from .data_proc import recursion_trans_php_file_opcode

FAST_LOAD = False  # 是否要快速加载w2v


def most_similar(w2v_model, word, topn=10):
    try:
        similar_words = w2v_model.wv.most_similar(word, topn=topn)
    except:
        print(word, "not found in Word2Vec model!")
    return similar_words


def train_word2vec_only(op_dir):
    '''
    仅训练w2v，不向量化opcode
    '''
    y = []
    payloads = []
    payloads_seged = []
    lens = []

    # 加载训练数据
    with open(op_dir) as fr:
        while (1):
            payload = fr.readline()
            if (payload == '\r\n' or payload == '\n' or payload == '\r'):
                continue
            if (not payload):
                good_len = len(payloads)
                print(f'[+num] the num of the word2vec train set is {good_len}' )
                break
            payload = payload.strip()
            payloads.append(payload)


    # payload分段
    for payload in payloads:
        tempseg = payload.split(' ')
        if (tempseg == []):
            print(payload[:100])
        payloads_seged.append(tempseg)
        lens.append(len(tempseg))

    # 储存长度信息
    with open(lens_dir, 'wb') as f:
        pickle.dump(lens, f)

    # Word2vec模型训练

    model = Word2Vec(payloads_seged,
                     size=embedding_size,
                     iter=iter_num,
                     sg=1,
                     min_count=min_num,
                     max_vocab_size=max_voc)

    model.save(vec_dir)

def train_word2vec():
    '''
    读取opcode，训练word2vec

    将opcode向量化储存为.npy
    '''

    y = []
    payloads = []
    payloads_seged = []
    lens = []

    # 加载非恶意数据
    with open(good_ops_dir) as fr:
        while (1):
            payload = fr.readline()
            if (payload == '\r\n' or payload == '\n' or payload == '\r'):
                continue
            if (not payload):
                good_len = len(payloads)
                print('[num] 非恶意数据量为：', good_len)
                break
            payload = payload.strip()
            payloads.append(payload)
            y.append(0)

    # 加载恶意数据
    with open(bad_ops_dir) as fr:
        while (1):
            payload = fr.readline()
            if (payload == '\r\n' or payload == '\n' or payload == '\r'):
                continue
            if (not payload):
                print('[num] 恶意数据量为：', len(payloads) - good_len)
                break
            payload = payload.strip()
            payloads.append(payload)
            y.append(1)

    # 储存y
    y = np.array(y)
    np.save(y_train_dir, y)

    # payload分段
    for payload in payloads:
        tempseg = payload.split(' ')
        if (tempseg == []):
            print(payload[:100])
        payloads_seged.append(tempseg)
        lens.append(len(tempseg))

    # 储存长度信息
    with open(lens_dir, 'wb') as f:
        pickle.dump(lens, f)

    # Word2vec模型训练

    if not FAST_LOAD:

        model = Word2Vec(payloads_seged,
                         size=embedding_size,
                         iter=iter_num,
                         sg=1,
                         min_count=min_num,
                         max_vocab_size=max_voc)

        model.save(vec_dir)

    else:
        model = Word2Vec.load(vec_dir)

    # 向量化数据集

    x = []
    tempvx = []
    for payload in payloads_seged:
        for word in payload:
            try:
                tempvx.append(model.wv.get_vector(word))
            except KeyError as e:
                tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位
        tempvx = np.array(tempvx)
        if (tempvx.shape[0] == 0):
            print(payload[:100])
        x.append(tempvx)
        # print(tempvx.shape)
        tempvx = []

    # 字符串向量长度填充
    lenth = time_step
    for i in range(y.shape[0]):
        if (x[i].shape[0] < lenth):
            try:
                x[i] = np.pad(x[i], ((0, lenth - x[i].shape[0]), (0, 0)),
                              'constant',
                              constant_values=0)
            except ValueError as e:
                print(i)
                print(x[i].shape)
                print(x[i])
                exit()
        elif (x[i].shape[0] > lenth):
            x[i] = x[i][0:lenth]
    x = np.array(list(x))
    # print(x.shape)
    np.save(x_train_dir, x)  # 储存向量化数据集

    # 验证w2v模型质量

    # print(model.wv.vocab)
    # print(most_similar(model, 'EXT_STMT'))
    # for i in most_similar(model, 'JMPZ', 5):
    #     print(i)
    # print(model.wv['EXT_STMT'].shape)


# def vectorize_payloads(payloads_dir,
#                        save_x_dir,
#                        save_y_dir,
#                        write_op_dir='data/obfuscated_op.csv',
#                        is_bad=True,
#                        wr_result=False,
#                        result_file='result/ob_shell_num.txt'):
#     '''
#     使用训练好的word2vec模型向量化payloads_dir子目录下的所有php文件，
    
#     储存至save_x_dir,save_y_dir
#     '''

#     recursion_trans_php_file_opcode(payloads_dir, write_op_dir)  # 递归处理子目录的数据

#     payloads = []
#     payloads_seged = []
#     lens = []

#     # 加载数据
#     with open(write_op_dir) as fr:
#         while (1):
#             payload = fr.readline()
#             if (payload == '\r\n' or payload == '\n' or payload == '\r'):
#                 continue

#             if (not payload):
#                 break

#             payload = payload.strip()
#             payloads.append(payload)

#     # payloads = list(set(payloads))  # 去重
#     print('[num] 数据量为：', len(payloads))

#     if wr_result:
#         with open(result_file,'a') as fw:
#             print('[num] 本轮混淆shell数据量为：', len(payloads),file=fw)

#     if is_bad:
#         y = [1] * len(payloads)
#     else:
#         y = [0] * len(payloads)

#     y = np.array(y)
#     np.save(save_y_dir, y)

#     for payload in payloads:
#         tempseg = payload.split(' ')
#         if (tempseg == []):
#             print(payload[:100])
#         payloads_seged.append(tempseg)
#         lens.append(len(tempseg))

#     model = Word2Vec.load(vec_dir)

#     # 向量化数据集

#     x = []
#     tempvx = []
#     for payload in payloads_seged:
#         for word in payload:
#             try:
#                 tempvx.append(model.wv.get_vector(word))
#             except KeyError as e:
#                 tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位
#         tempvx = np.array(tempvx)
#         if (tempvx.shape[0] == 0):
#             print(payload[:100])
#         x.append(tempvx)
#         # print(tempvx.shape)
#         tempvx = []

#     # 字符串向量长度填充
#     lenth = time_step
#     for i in range(y.shape[0]):
#         if (x[i].shape[0] < lenth):
#             try:
#                 x[i] = np.pad(x[i], ((0, lenth - x[i].shape[0]), (0, 0)),
#                               'constant',
#                               constant_values=0)
#             except ValueError as e:
#                 print(i)
#                 print(x[i].shape)
#                 print(x[i])
#                 exit()
#         elif (x[i].shape[0] > lenth):
#             x[i] = x[i][0:lenth]
#     x = np.array(list(x))
#     # print(x.shape)
#     np.save(save_x_dir, x)  # 储存向量化数据集


# def op2vector(op_dir):
#     '''
#     将opcode文件转化为向量，并返回变量
#     '''
#     with open(op_dir,'rb') as fr:
#         payloads=pickle.load(fr)
#         print('[num] 数据量为：', len(payloads))


#     x = []
#     tempvx = []
#     payloads_seged = []
#     model = Word2Vec.load(vec_dir)

#     for payload in payloads:
#         tempseg = payload.split(' ')
#         if (tempseg == []):
#             print(payload[:100])
#         payloads_seged.append(tempseg)

#     for payload in payloads_seged:
#         for word in payload:
#             try:
#                 tempvx.append(model.wv.get_vector(word))
#             except KeyError as e:
#                 tempvx.append(np.zeros((embedding_size)))  # 若不在字典中，则输入0占位
#         tempvx = np.array(tempvx)
#         if (tempvx.shape[0] == 0):
#             print(payload[:100])
#         x.append(tempvx)
#         # print(tempvx.shape)
#         tempvx = []

#     # 字符串向量长度填充
#     lenth = time_step
#     for i in range(len(x)):
#         if (x[i].shape[0] < lenth):
#             try:
#                 x[i] = np.pad(x[i], ((0, lenth - x[i].shape[0]), (0, 0)),
#                               'constant',
#                               constant_values=0)
#             except ValueError as e:
#                 print(i)
#                 print(x[i].shape)
#                 print(x[i])
#                 exit()
#         elif (x[i].shape[0] > lenth):
#             x[i] = x[i][0:lenth]
#     x = np.array(list(x))
    
#     return x
    
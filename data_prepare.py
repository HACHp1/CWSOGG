from discriminator.data_proc import *
from discriminator.attention_tf2 import *
from discriminator.word2vec import *

from vectorize import vectorize_php_dataset


from utils import phproot,phpgoodroot

'''
1. 训练word2vec
↓
2. PHP->opcode+cfg解析->pickle（分为测试集和训练集）
'''


if __name__ == '__main__':

    sep_data(phpgoodroot, phproot, 0.2)  # 分割训练集测试集
    trans_php_file_opcode('data/train_op_w2v.csv')  # php转化为opcode
    train_word2vec_only('data/train_op_w2v.csv')  # 使用opcode训练word2vec
    vectorize_php_dataset('data/data_sep/train/train.txt',
                          x_train_dir, y_train_dir)  # 向量化训练集
    vectorize_php_dataset('data/data_sep/test/test.txt',
                          x_test_dir, y_test_dir)  # 向量化测试集

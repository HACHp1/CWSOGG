from utils import *
from .attention_tf2 import *
from .word2vec import *

from vectorize import php_one_vectorize

w2v_model = Word2Vec.load(vec_dir)

model = build_model()
model.load_weights('model/attention_weights.keras')  # 载入模型参数

'''
向量化函数
传入：单个webshell的源代码
传出：向量化的结果
'''


def tovector(payload):

    temp_file = 'tmp/' + hex(random.randint(
        0, 1000000000000))[2:] + '.php'

    with open(temp_file, 'w') as fw:
        fw.write(payload)

    vec = php_one_vectorize(temp_file, w2v_model)

    try:
        os.remove(temp_file)
    except FileNotFoundError:
        pass

    return vec


def attention_detect(x_test):
    pred_y = model.predict(x_test)
    return np.argmax(pred_y)


def attention_logit(x_test):
    modelLock.acquire() # 获取模型锁
    pred_y = model.predict(x_test, batch_size=x_test.shape[0])
    modelLock.release() # 释放模型锁
    return pred_y


if __name__ == '__main__':
    a = '<?php eval($_GET[123]); ?>'
    a_vec = tovector(a)
    # print(a_vec.shape)
    # print(a_vec)
    print(attention_detect(a_vec.reshape(
        1, time_step * embedding_size+low_feature_dim)))

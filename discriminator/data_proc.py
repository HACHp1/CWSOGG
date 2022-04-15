'''
数据预处理
将黑白PHP文件转化为操作符序列并储存到txt中

'''

from utils import *
import os
import pickle


def recursion_trans_php_file_opcode(dir, write_dir):
    '''
    递归遍历目标文件夹内的所有的PHP文件，
    将其转化为操作符序列并写入到文件中
    '''
    print('开始生成 {} 路径中的PHP的opcode操作码文件'.format(dir))

    with open(write_dir, 'w') as fw:
        for root, dirs, files in os.walk(dir):
            for filename in files:
                if '.php' in filename or '.phtml' in filename:
                    try:
                        full_path = os.path.join(root, filename)
                        file_content = load_php_opcode(full_path)
                        if (file_content == ''):  # 空文件或读取失败时跳过该文件
                            continue
                        fw.write(file_content + '\n')
                    except:
                        continue


def wash_ops():
    '''
    opcode 去重
    '''
    payloads = []

    # 加载非恶意数据
    with open(good_ops_dir_unwash) as fr:
        while (1):
            payload = fr.readline()
            if (payload == '\r\n' or payload == '\n' or payload == '\r'):
                continue
            if (not payload):
                break
            payload = payload.strip()
            payloads.append(payload)

    payloads = list(set(payloads))
    print('good num after washed:', len(payloads))
    with open(good_ops_dir, 'w') as fw:
        for payload in payloads:
            fw.write(payload + '\n')

    payloads = []
    # 加载恶意数据
    with open(bad_ops_dir_unwash) as fr:
        while (1):
            payload = fr.readline()
            if (payload == '\r\n' or payload == '\n' or payload == '\r'):
                continue
            if (not payload):
                break
            payload = payload.strip()
            payloads.append(payload)

    payloads = list(set(payloads))
    print('bad num after washed:', len(payloads))
    with open(bad_ops_dir, 'w') as fw:
        for payload in payloads:
            fw.write(payload + '\n')


# def prepare_data():
#     '''
#     提取good和bad的opcode至各自的路径（good_dir和bad_dir）

#     opcode去重
#     '''
#     recursion_trans_php_file_opcode(good_dir, good_ops_dir_unwash)
#     recursion_trans_php_file_opcode(bad_dir, bad_ops_dir_unwash)
#     wash_ops()


def op_train_test_spilt(test_size=0.2):
    '''
    将opcode划分为train、test集合，并储存为文件
    '''
    payloads = []
    y = []
    # 加载非恶意数据
    with open(good_ops_dir) as fr:
        while (1):
            payload = fr.readline()
            if (payload == '\r\n' or payload == '\n' or payload == '\r'):
                continue
            if (not payload):
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
                break
            payload = payload.strip()
            payloads.append(payload)
            y.append(1)

    payloads, y = shuffle_data(payloads, y)

    test_num = int(len(y) * 0.2)

    payloads_test = payloads[:test_num]
    y_test = y[:test_num]

    payloads_train = payloads[test_num:]
    y_train = y[test_num:]

    with open('data/train_test_split/test/ops.pkl', 'wb') as fw:
        pickle.dump(payloads_test, fw)

    with open('data/train_test_split/test/y.pkl', 'wb') as fw:
        pickle.dump(y_test, fw)

    with open('data/train_test_split/train/ops.pkl', 'wb') as fw:
        pickle.dump(payloads_train, fw)

    with open('data/train_test_split/train/y.pkl', 'wb') as fw:
        pickle.dump(y_train, fw)


if __name__ == '__main__':
    # prepare_data()
    pass

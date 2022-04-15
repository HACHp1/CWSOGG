from threading import TIMEOUT_MAX
from .shell_enc_fun import *
from func_timeout import func_set_timeout
from utils import ENCODE_TIMEOUT

'''
混淆webshell的加工类
目标函数为 min webshell检测器的恶意输出
每次都要用原webshell进行重新编码，比较麻烦
'''


class Shell_encoder():

    # --函数列表--

    # 可重复编码的函数
    n_encode_li = [
        parse_str_encode,
        destruct_encode,
        null_encode,
        quote_encode,
        quotenull_encode,
        add_a_math,
        add_a_echo,
        xor_encode,
        not_encode
    ]

    # 只能编码一次的函数
    once_encode_li = [
        extract_encode,
        array_map_encode,
        call_user_func_array_encode,
        call_user_func_encode,
        
    ]

    # 所有编码函数
    all_encode_li = n_encode_li+once_encode_li

    orig_shell = ''  # 储存原始webshell

    # 当前编码后的webshell列表
    # shell_li = []

    n_encode_len = 0  # 可重复编码的长度

    def __init__(self, orig_shell):
        self.orig_shell = orig_shell
        self.n_encode_len = len(self.n_encode_li)


    @func_set_timeout(ENCODE_TIMEOUT)
    def code_shell_encode(self, enc_code):
        '''
        Encode the webshell according to the enc_code ,return the encoded webshell. 

        将遗传编码转化为shell混淆操作，返回混淆后的shell

        包括0编码的处理、只能编码一次的函数的处理
        '''
        temp_shell = self.orig_shell
        for i in range(len(enc_code)):
            enc_op = int(enc_code[i]) # 转化为int
            if enc_op != 0:
                enc_op = enc_op-1  # 跳过0的不变操作

                if enc_op < self.n_encode_len:  # 可重复编码
                    try:
                        temp_shell = self.all_encode_li[enc_op](temp_shell)
                    except Exception as e:
                        # print('[e] '+str(e) + ' error when try to encode the shell:\nmethod:\n{}\nshell:\n{}'.format(
                        #     self.all_encode_li[enc_op],
                        #     temp_shell)
                        # )
                        # exit()
                        continue
                        

                else:  # 不可重复编码
                    if enc_code[i] in enc_code[:i]:  # 如果已经编码过
                        continue
                    else:  # 未编码过
                        try:
                            temp_shell = self.all_encode_li[enc_op](temp_shell) ####
                        except Exception as e:
                            print('[e] '+str(e) + ' error when try to encode the shell:\nmethod:\n{}\nshell:\n{}'.format(
                                self.all_encode_li[enc_op],
                                temp_shell[:100])
                            )
                            exit()

            else:  # 0操作为不编码
                continue

        return temp_shell

    def get_enc_fun(self, code):
        '''
        return the encode function according to the index

        根据索引返回编码函数
        '''
        return self.all_encode_li[code-1]

    def get_fun_num(self):
        '''
        return the number of all encode methods 

        返回所有编码的种类数
        '''
        return len(self.all_encode_li)

    def get_n_fun_num(self):
        '''
        return the number of encode methods that can be used more than once

        返回可重复编码的种类数
        '''
        return self.n_encode_len


'''
免杀过程：
遗传编码：
1,2,6,...,2,（n为0到编码种数的整数） 每一个整数代表一种编码，0代表不编码

'''

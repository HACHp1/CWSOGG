from shellgenerator.orig_shells_generator import orig_shell_li
from shellgenerator.orig_shells import orig_shell_li as train_orig_shell_li
from shellgenerator.ga_bypass_att_train import ga_bypass_att_once


from utils import *
from shellgenerator.Shell_encoder import Shell_encoder
from discriminator.detect_attention_tf2 import *

from func_timeout import FunctionTimedOut, func_set_timeout
import time


model.load_weights('model/attention_weights.keras')  # 初始化为最原始的模型权重

VS_TIMEOUT = 120 * 60  # stop timeout task by seconds

opt_time = 5

rand_success = 0
ga_success = 0


rand_scores = []
ga_scores = []


@func_set_timeout(VS_TIMEOUT)
def time_task_ga():
    global ga_success, ga_scores

    my_encoder = Shell_encoder('')

    epoch = 0
    while 1:

        # 获取N个GA优化的混淆组合
        ga_code = []
        for _ in range(opt_time):
            orig_shell = orig_shell_li[_ % len(orig_shell_li)]
            ga_code.append(ga_bypass_att_once(
                orig_shell, max_encode_time, NIND, MAX_GEN))

        # 对所有训练数据进行混淆
        for v_shell in train_orig_shell_li:

            my_encoder.orig_shell = v_shell

            for v_code in ga_code:
                ga_shell = my_encoder.code_shell_encode(v_code)

                try:
                    ga_score = attention_logit(tovector(ga_shell).reshape(
                        1, time_step * embedding_size+low_feature_dim))[0][0]
                except Exception:
                    ga_score = 0

                ga_scores.append(ga_score)

                # 统计
                if ga_score > 0.5:
                    ga_success += 1

        epoch += 1


@func_set_timeout(VS_TIMEOUT)
def time_task_rand():
    global rand_success, rand_scores

    my_encoder = Shell_encoder('')
    encode_type_num = my_encoder.get_fun_num()

    epoch = 0
    while 1:

        # 获取N个随机混淆组合
        rand_code = []
        for _ in range(opt_time):
            rand_code.append(np.random.randint(0, encode_type_num, (12)))

        # 对所有训练数据进行混淆
        for v_shell in train_orig_shell_li:

            my_encoder.orig_shell = v_shell

            for v_code in rand_code:
                rand_shell = my_encoder.code_shell_encode(v_code)

                try:
                    rand_score = attention_logit(tovector(rand_shell).reshape(
                        1, time_step * embedding_size+low_feature_dim))[0][0]
                except Exception:
                    rand_score = 0

                rand_scores.append(rand_score)

                # 统计
                if rand_score > 0.5:
                    rand_success += 1

        epoch += 1


if __name__ == '__main__':

    time1 = time.time()

    try:
        time_task_ga()
    except FunctionTimedOut:
        pass

    time2 = time.time()

    try:
        time_task_rand()
    except FunctionTimedOut:
        pass

    time3 = time.time()

    with open('result/vs_att.txt', 'w') as fw:
        print('rand_success:', rand_success, file=fw)
        print('ga_success:', ga_success, file=fw)
        print('Time:', time2-time1, time3-time2, file=fw)

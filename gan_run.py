import os
import datetime


models = [
    'att',
    'lstm',
    'cnn',
    'dnn',
]

# 清空记录
with open('result/time.txt', 'w') as fw:
    pass

for mod in models:

    starttime = datetime.datetime.now()

    print("python shell_gan_ga_"+mod+".py")
    os.system("python shell_gan_ga_"+mod+".py")

    endtime = datetime.datetime.now()

    with open('result/time.txt', 'a') as fw:
        print(mod+': ', end='', file=fw)
        print(starttime.strftime("%Y_%m_%d_%H_%M_%S")+', ' +
              endtime.strftime("%Y_%m_%d_%H_%M_%S"), file=fw)

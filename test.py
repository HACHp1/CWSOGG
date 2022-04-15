
from shellgenerator.Shell_encoder import Shell_encoder

import pickle

# with open('bin/ob_ga_att_code.pkl','rb') as fr:
#     enc=pickle.load(fr)

# print(enc)


for i in range(9,10):
    print(1)

exit()

enc=['7', '12', '7', '9', '11', '9', '11', '4', '3', '7', '11', '0']
enc=['2', '5', '12', '6', '5', '9', '9', '12', '5', '11', '5', '11']
enc=['2']

shell='4a2175db3bb30db21fbb681b8c4fe840.php'


orig_shell=open('data/bad_collect_all/'+shell).read()
web_encoder = Shell_encoder(orig_shell)

ob_shell = web_encoder.code_shell_encode(enc)
# print(ob_shell)
with open('tmp/temp','w') as fw:
    fw.write(ob_shell)
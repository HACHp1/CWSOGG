'''
# basic blocks store format:[ # a phpfile is parsed to several segments(E.g. functions, classes),every segment is transformed into a CFG
#                               [ # a CFG has several blocks
#                                   [[x,x,x,x,x,x],[x,x,x,x,x,x]],... # a block has several nodes, every node contains several instructions 
#                                                           ]]
#                      [x,x,x,x,x,x] is [php_code_line, opcode, fetch, ext, return, operands]
# every bb is stored one by one so you can easily get the bb by its index                                                             ] 

# edges store format :[ # a phpfile is parsed to several CFGs
#                       [ # every CFG has a edgs group
#                          [x,x],...],...] 
# the first 'x' of the [x,x] is the index of the start bb of the edge 
# the second 'x' is the index of the target bb of the edge
'''


import subprocess


def load_php_opcode(phpfilename):
    """
    Return the php opcode.

    获取php opcode 信息；提取一个php文件为opcode操作符连接的句子
    """
    try:

        output = subprocess.check_output(
            ['php', '-dvld.active=1', '-dvld.execute=0', '-dvld.verbosity=0',
             '-dvld.skip_prepend=1', '-dvld.skip_append=1', phpfilename],
            stderr=subprocess.STDOUT)
        return output

    except Exception as e:
        # print('[Error] ', phpfilename, ' Error: ', e)
        # exit()
        return b""  # 未读取成功或没有任何操作符时


def first_split(str_target, str_to_find):
    '''
    split the str_target by the first find of str_to_find
    '''
    pos = str_target.find(str_to_find)
    return str_target[:pos], str_target[pos:]


vld_header = b'''-------------------------------------------------------------------------------------\n'''
vld_end = b'''\n\n'''

line_offset = 0
line_len = 5

op_offset = 19
op_len = 29

fetch_offset = 48
fetch_len = 15

ext_offset = 63
ext_len = 3

return_offset = 68
return_len = 8

operands_offset = 76

jmp_op = [b'JMP', b'JMPZ', b'JMPNZ', b'JMPZNZ', b'JMPZ_EX', b'JMPNZ_EX']
exit_op = [b'EXIT', b'RETURN']


def get_cfgs(vld_output):
    '''
    input: vld output of a php file
    return: [cfgs],[edgs]
    '''
    cfgs = []
    edgs_list = []

    while vld_header in vld_output:
        _, vld_output = first_split(vld_output, vld_header)
        vld_out, vld_output = first_split(
            vld_output[len(vld_header):], vld_end)

        out_li = []
        for line in vld_out.split(b'\n'):
            tmp_li = []
            # get line's php source code line num
            tmp_li.append(line[line_offset:line_offset+line_len].strip())
            # get line's opcode
            tmp_li.append(line[op_offset:op_offset+op_len].strip())

            # get line's fetch
            tmp_li.append(line[fetch_offset:fetch_offset+fetch_len].strip())
            # get line's ext
            tmp_li.append(line[ext_offset:ext_offset+ext_len].strip())
            tmp_li.append(line[return_offset:return_offset +
                               return_len].strip())  # get line's retrun
            # get line's ext operands
            tmp_li.append(line[operands_offset:].strip())

            out_li.append(tmp_li)

        b_entry = set()  # store the op indexes of the basic block entries

        edgs = []

        for i in range(len(out_li)):
            if out_li[i][1] in jmp_op:
                operand = out_li[i][5]
                tg_start = operand.find(b'->')
                target = operand[tg_start+2:]

                b_entry.add(int(target))
                b_entry.add(i+1)

                edgs.append([i, int(target)])
                if out_li[i][1] != b'JMP':
                    edgs.append([i, i+1])

        b_entry = list(b_entry) 
        b_entry.sort() # must sort before link the bb

        bbs = []

        bbs_edgs = []

        op2bb_ind = []

        offset = 0
        bb_ind = 0
        for i in b_entry: # b_entry must be sorted first
            op2bb_ind = op2bb_ind + ([bb_ind]*(i-offset))
            bbs.append(out_li[offset:i])
            offset = i
            bb_ind += 1

        op2bb_ind = op2bb_ind + ([bb_ind]*(len(out_li)-offset))
        bbs.append(out_li[offset:])

        for edg in edgs:
            bbs_edgs.append((op2bb_ind[edg[0]], op2bb_ind[edg[1]]))

        cfgs.append(bbs)
        edgs_list.append(bbs_edgs)

    return cfgs, edgs_list


def get_cfgs_fromfile(phpfile):
    '''
    input: a php file diretory
    return: [cfgs],[edgs]
    '''
    vld_output = load_php_opcode(phpfile)
    return get_cfgs(vld_output)

if __name__ == '__main__':
    phpfile = 'phpcode/hellowd.php'
    phpfile = 'phpcode/jmp.php'
    phpfile = 'E:/HONKER/python_program/webshell_gan_all2/data/phply_test/1.php'

    vld_output = load_php_opcode(phpfile)
    bbs, edgs = get_cfgs(vld_output)


    for i in range(len(edgs[0])):
        print(bbs[0][edgs[0][i][0]][0],'->',bbs[0][edgs[0][i][1]][0])
        # print(len(bbs[0]))

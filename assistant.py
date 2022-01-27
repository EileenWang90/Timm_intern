import torch
import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d
###############################################################################
layer_type=(nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, SelectAdaptivePool2d, nn.Linear)
### forward_pre_hook 
def forward_pre_hook(module, inputdata):
    '''把这层的输出拷贝到features中'''
    # print('HOOK before cast:', type(inputdata),len(inputdata), inputdata[0].shape) # fc  HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 2048])
    # print(inputdata[0][0])
    print('HOOK before cast:', type(inputdata),len(inputdata), inputdata[0].shape, inputdata[0][0][0][0].shape)  #HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 3, 160, 160]) torch.Size([160])
    print('HOOK before cast:', inputdata[0][0][0][0])
    inputdata = cast_fp32_tf32(inputdata)
    print('HOOK after cast:', type(inputdata),len(inputdata), inputdata[0].shape, inputdata[0][0][0][0].shape)  #HOOK OUT: <class 'tuple'> 1 torch.Size([256, 3, 160, 160])
    print('HOOK after cast:', inputdata[0][0][0][0])
    # print('HOOK after cast:', inputdata[0][0])


def print_hook(module, inputdata):
    for index, (name, child) in enumerate(module.named_children()):
        # print(index, name, child)
        if isinstance(child, layer_type):  # manual
            print('###', index, name, type(child), child)
            # tmpstr=module + '.'+ child
            # eval()
        elif isinstance(child, nn.Sequential):
            print('Sequential',index, name, type(child), child)
        else:
            # print_hook(child, inputdata)
            print('ooo', index, name, type(child), child)
    # for idx,(name,m) in enumerate(module.named_parameters()):
    #     print(idx,"-",name,m.size())

    submodules=[]
    if isinstance(submodules, str):
        submodules = [submodules]

    named_modules = submodules
    submodules = [module.get_submodule(m) for m in submodules]

    if not len(submodules):
        named_modules, submodules = list(zip(*module.named_children()))

    for name, m in zip(named_modules, submodules):
        print(idx,"-",name,m)

    # for idx,(name,m) in enumerate(module._modules.items()):
    #     print(idx,"-",name,m)

    input('Here to stop!')



def cast_fp32_tf32(input, mode="rne"):  # type(input)=tuple
    # print('input:',input[0][0][0][0])
    in_int = input[0].view(torch.int32)
    # print('in_int:',in_int[0][0][0])
    mask = torch.tensor(0xffffe000, dtype=torch.int32)
 
    if mode == "trunc":
        output = torch.bitwise_and(in_int, mask).view(torch.float)
        return (output,)
 
    in_size = input[0].size()
    in_round = torch.zeros(in_size, dtype=torch.int32)
 
    # we don't round nan and inf
    do_round = torch.ones(in_size, dtype=torch.bool)
    nan_mask = torch.tensor(0x7f800000, dtype=torch.int32)
    do_round = torch.where(torch.bitwise_and(in_int, nan_mask) == 0x7f800000, False, True)
 
    # perform round nearest tie even
    sr = torch.tensor(13, dtype=torch.int32)
    one = torch.tensor(1, dtype=torch.int32)
    point5 = torch.tensor(0x00000fff, dtype=torch.int32)
    fixup = torch.bitwise_and(torch.bitwise_right_shift(in_int, sr), one)
    in_round = in_int + point5 + fixup
    in_round = torch.where(do_round, in_round, in_int)
 
    mask = torch.tensor(0xffffe000, dtype=torch.int32)
 
    output = torch.bitwise_and(in_round, mask).view(torch.float)
    # print('output:',output[0][0][0])
 
    return (output,)
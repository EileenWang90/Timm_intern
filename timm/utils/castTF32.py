import torch
import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d

###############################################################################
layer_type=(nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, SelectAdaptivePool2d, nn.Linear)
def forward_pre_hook(module, inputdata):
    inputdata = cast_fp32_tf32_tuple(inputdata)
    return inputdata

def forward_pre_hook_verbose(module, inputdata): 
    # print('HOOK before cast:', type(inputdata),len(inputdata), inputdata[0].shape, inputdata[0][0][0][0].shape)  #HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 3, 160, 160]) torch.Size([160])
    # print('HOOK before cast:', inputdata[0][0][0][0])
    print('forward_pre_hook before cast:', type(inputdata),len(inputdata),inputdata[0].shape)
    print('forward_pre_hook before cast:', inputdata[0][0])
    inputdata = cast_fp32_tf32_tuple(inputdata)
    print('forward_pre_hook after cast:', type(inputdata),len(inputdata),inputdata[0].shape)
    print('forward_pre_hook after cast:', inputdata[0][0])
    # print('HOOK after cast:', type(inputdata),len(inputdata), inputdata[0].shape, inputdata[0][0][0][0].shape)  #HOOK OUT: <class 'tuple'> 1 torch.Size([256, 3, 160, 160])
    # print('HOOK after cast:', inputdata[0][0][0][0])
    return inputdata

def forward_hook(module, inputdata, outputdata):
    # print('forward_hook before cast:', type(inputdata),len(inputdata),inputdata[0].shape) # HOOK before cast: <class 'tuple'> 2 torch.Size([128, 1000])
    # print('forward_hook before cast:', inputdata[0][0])
    # input('---stop here---')
    # print('HOOK before cast:', type(outputdata),outputdata) # HOOK before cast: <class 'torch.Tensor'> tensor(0.7046, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
    outputdata = cast_fp32_tf32(outputdata).requires_grad_(True)
    # print('HOOK after cast:', type(outputdata),outputdata) # HOOK after cast: <class 'torch.Tensor'> tensor(0.7046, device='cuda:0')
    return outputdata


def backward_hook(module, grad_input, grad_output):
    # print(module)
    # if (grad_input is None) or (grad_output is None):
    #     print('Meet NoneType.')
    # else:
    #     print('    input:', len(grad_input), grad_input[0].shape)
    #     print('    output:', len(grad_output), grad_output[0].shape)
    # print('HOOK before cast:', type(grad_input),len(grad_input), grad_input[0].shape)  #HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 3, 160, 160]) torch.Size([160])
    # print('before grad_input:', grad_input[0][0])
    # print('HOOK before cast:', type(grad_output),len(grad_output), grad_output[0].shape) 
    # print('before grad_output:', grad_output[0][0])
    grad_input = cast_fp32_tf32_tuple(grad_input)
    # print('HOOK after cast:', type(grad_input),len(grad_input), grad_input[0].shape)  #HOOK OUT: <class 'tuple'> 1 torch.Size([256, 3, 160, 160])
    # print('after grad_input:', grad_input[0][0])
    # print('HOOK after cast:', type(grad_output),len(grad_output), grad_output[0].shape) 
    # print('after grad_output:', grad_output[0][0])
    # input('Here to stop.')
    return grad_input 


def register_hook(grad):
    # print('    grad:', len(grad), grad.shape)
    # print('before', grad)
    grad = cast_fp32_tf32(grad)
    # print('    grad:', len(grad), grad.shape)
    # print('after', grad)
    # input('---stop here---')
    return grad


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

@torch.no_grad()
def cast_fp32_tf32(input, mode="rne"):  # type(input) != tuple
    in_int = input.view(torch.int32)
    mask = torch.tensor(0xffffe000, dtype=torch.int32)
 
    if mode == "trunc":
        output = torch.bitwise_and(in_int, mask).view(torch.float)
        return output
 
    in_size = input.size()
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
 
    return output

@torch.no_grad()
def cast_fp32_tf32_float(input, mode="rne"):  # type(input) != tuple
    input = torch.tensor(input)
    output = cast_fp32_tf32(input, mode=mode)
    output = float(output)
    return output

@torch.no_grad()
def cast_fp32_tf32_tuple(input_tuple, mode="rne"):  # type(input)=tuple
    assert type(input_tuple) == tuple
    output_list=[]
    for input in input_tuple: 
        output = cast_fp32_tf32(input, mode=mode)
        output_list.append(output.requires_grad_(True))
    return tuple(output_list)

# @torch.no_grad()
# def cast_fp32_tf32_tuple(input_tuple, mode="rne"):  # type(input)=tuple
#     assert type(input_tuple) == tuple

#     output_list=[]
#     for input in input_tuple: 
#         # print('input:',input[0][0][0][0])
#         in_int = input.view(torch.int32)
#         # print('in_int:',in_int[0][0][0])
#         mask = torch.tensor(0xffffe000, dtype=torch.int32)
    
#         if mode == "trunc":
#             output = torch.bitwise_and(in_int, mask).view(torch.float)
#             return (output,)
    
#         in_size = input.size()
#         in_round = torch.zeros(in_size, dtype=torch.int32)
    
#         # we don't round nan and inf
#         do_round = torch.ones(in_size, dtype=torch.bool)
#         nan_mask = torch.tensor(0x7f800000, dtype=torch.int32)
#         do_round = torch.where(torch.bitwise_and(in_int, nan_mask) == 0x7f800000, False, True)
    
#         # perform round nearest tie even
#         sr = torch.tensor(13, dtype=torch.int32)
#         one = torch.tensor(1, dtype=torch.int32)
#         point5 = torch.tensor(0x00000fff, dtype=torch.int32)
#         fixup = torch.bitwise_and(torch.bitwise_right_shift(in_int, sr), one)
#         in_round = in_int + point5 + fixup
#         in_round = torch.where(do_round, in_round, in_int)
    
#         mask = torch.tensor(0xffffe000, dtype=torch.int32)
    
#         output = torch.bitwise_and(in_round, mask).view(torch.float).requires_grad_(True) # add .requires_grad_(True)
#         # print('output:',output[0][0][0])
#         output_list.append(output)
 
#     return tuple(output_list)


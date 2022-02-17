import torch
import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d
import time

@torch.no_grad()
def cast_fp32_tf32(input, mode="rne"):  # type(input) != tuple
    input = input.view(torch.int32)
    mask = torch.tensor(0xffffe000, dtype=torch.int32)
 
    if mode == "trunc":
        input = torch.bitwise_and(input, mask).view(torch.float)
        return input
 
    in_size = input.size()
    in_round = torch.zeros(in_size, dtype=torch.int32)
 
    # we don't round nan and inf
    do_round = torch.ones(in_size, dtype=torch.bool)
    nan_mask = torch.tensor(0x7f800000, dtype=torch.int32)
    do_round = torch.where(torch.bitwise_and(input, nan_mask) == 0x7f800000, False, True)
 
    # perform round nearest tie even
    sr = torch.tensor(13, dtype=torch.int32)
    one = torch.tensor(1, dtype=torch.int32)
    point5 = torch.tensor(0x00000fff, dtype=torch.int32)
    fixup = torch.bitwise_and(torch.bitwise_right_shift(input, sr), one)
    in_round = input + point5 + fixup
    input = torch.where(do_round, in_round, input)
 
    # mask = torch.tensor(0xffffe000, dtype=torch.int32)
 
    input = torch.bitwise_and(input, mask).view(torch.float)
 
    return input


@torch.no_grad()
def cast_fp32_tf32_storage(input, mode="rne"):  # # modify data buffer, thus no return 
    in_int = input.view(torch.int32) #share same storage
    mask = torch.tensor(0xffffe000, dtype=torch.int32)
 
    if mode == "trunc":
        output = torch.bitwise_and(in_int, mask).view(torch.float)
    else:  #mode == "rne"
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
        # in_round.where_(do_round, in_int)
        output = torch.bitwise_and(in_int, mask).view(torch.float).flatten()

    buffer_size = len(output)
    for index in range(buffer_size):
        input.storage()[index] = float(output[index])
    input.cuda()
    print('input in storage:', input)
        # return output


# @torch.no_grad()
# def cast_fp32_tf32_inplace(input, mode="rne"):  # type(input) != tuple
#     in_int = input.view(torch.int32) #share same storage
    
#     mask = torch.tensor(0xffffe000, dtype=torch.int32) # -8192 [torch.cuda.IntStorage of size 1] [torch.IntStorage of size 1] 
    
#     # print(input.device, in_int.device)
#     # print(mask.storage())
#     # print(input.data_ptr(), in_int.data_ptr())
#     # print(input.storage(), in_int.storage())  # cuda中的tensor用in_int.storage()就卡死？
    
#     if mode == "trunc":
#         in_int.bitwise_and_(mask)
#         # in_int = torch.bitwise_and(in_int, mask)
#         # return output
 
 
#     in_size = input.size()
#     in_round = torch.zeros(in_size, dtype=torch.int32)
 
#     # we don't round nan and inf
#     do_round = torch.ones(in_size, dtype=torch.bool)
#     nan_mask = torch.tensor(0x7f800000, dtype=torch.int32)
#     do_round = torch.where(torch.bitwise_and(in_int, nan_mask) == 0x7f800000, False, True)
 
#     # perform round nearest tie even
#     sr = torch.tensor(13, dtype=torch.int32)
#     one = torch.tensor(1, dtype=torch.int32)
#     point5 = torch.tensor(0x00000fff, dtype=torch.int32)
#     fixup = torch.bitwise_and(torch.bitwise_right_shift(in_int, sr), one)
#     in_round = in_int + point5 + fixup
#     in_round = torch.where(do_round, in_round, in_int)
#     # in_round.where_(do_round, in_int)
 
#     input = torch.bitwise_and(in_round, mask).view(torch.float)


@torch.no_grad()
def cast_fp32_tf32_inplace(input, mode="rne"):  # type(input) != tuple
    in_int = input.view(torch.int32)
    mask = torch.tensor(0xffffe000, dtype=torch.int32)

    in_size = input.size()
    in_storage = input.storage()
    output = torch.zeros(in_size, dtype=torch.float, device=input.device)
    # print(output.shape)
    output.set_(source=in_storage) #become one demension
    # print('input address: ', input.data_ptr())
    # print('output address0: ', output.data_ptr(),output.shape,input.size())
 
    if mode == "trunc":
        # output = torch.bitwise_and(in_int, mask).view(torch.float)
        in_int = in_int.view(-1)
        # print('len(in_int)', len(in_int))
        for index in range(len(in_int)):
            # print('before:',output.storage()[index])
            output.storage()[index] = float(torch.bitwise_and(in_int[index], mask).view(torch.float))
            # print('after:',output.storage()[index])
        # print('output address1: ', output.data_ptr())
        # print('input after cast: ',input.shape,input[0][0][0][0])
        # print('output after cast: ',output.shape,output[-160:])

        # return output
    else:
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
        print('input after cast: ',input.shape,input[0][0][0])
        print('output after cast: ',output.shape,output[0][0][0])
    
        # return output


@torch.no_grad()
def cast_fp32_tf32_inplaceV2(input, mode="rne"):  # type(input) != tuple
    mask = torch.tensor(0xffffe000, dtype=torch.int32)

    in_size = input.size()
    in_storage = input.storage()
    output = torch.zeros(in_size, dtype=torch.float, device=input.device)
    # print(output.shape)
    output.set_(source=in_storage) #become one demension
    in_int = output.view(torch.int32).view(in_size)
    # print('input address: ', input.data_ptr())
    # print('output address0: ', output.data_ptr(), output.shape)
    # print('in_int address0: ', in_int.data_ptr(), in_int.shape)
 
    if mode == "trunc":
        # output = torch.bitwise_and(in_int, mask).view(torch.float)
        # in_int = in_int.view(-1)
        # print('len(in_int)', len(in_int))
        # for index in range(len(in_int)):
        #     # print('before:',output.storage()[index])
        #     output.storage()[index] = float(torch.bitwise_and(in_int[index], mask).view(torch.float))
        #     # print('after:',output.storage()[index])
        in_int.bitwise_and_(mask)
        # print('output address1: ', output.data_ptr())
        # print('input after cast: ',input.shape,input[0][0][0][0])
        # print('output after cast: ',output.shape,output[-160:])

        # return output
    else:
        # in_round = torch.zeros(in_size, dtype=torch.int32)
    
        # # we don't round nan and inf
        # do_round = torch.ones(in_size, dtype=torch.bool)
        # nan_mask = torch.tensor(0x7f800000, dtype=torch.int32)
        # do_round = torch.where(torch.bitwise_and(in_int, nan_mask) == 0x7f800000, False, True)
    
        # perform round nearest tie even
        sr = torch.tensor(13, dtype=torch.int32)
        one = torch.tensor(1, dtype=torch.int32)
        point5 = torch.tensor(0x00000fff, dtype=torch.int32)
        fixup = torch.bitwise_and(torch.bitwise_right_shift(in_int, sr), one)
        in_int.add_(point5).add_(fixup) 
        # in_round = torch.where(do_round, in_round, in_int)

        in_int.bitwise_and_(mask)

        # print('output address1: ', output.data_ptr(), output.shape)
        # print('in_int address1: ', in_int.data_ptr(), in_int.shape)
    
        # return output


@torch.no_grad()
def cast_fp32_tf32_gpu(input, mode="rne"):  # type(input) != tuple
    device = input.device
    in_int = input.view(torch.int32)
    mask = torch.tensor(0xffffe000, device=device, dtype=torch.int32)
 
    if mode == "trunc":
        output = torch.bitwise_and(in_int, mask).view(torch.float)
        return output
 
    in_size = input.size()
    in_round = torch.zeros(in_size, device=device, dtype=torch.int32)
 
    # we don't round nan and inf
    do_round = torch.ones(in_size, device=device, dtype=torch.bool)
    nan_mask = torch.tensor(0x7f800000, device=device, dtype=torch.int32)
    do_round = torch.where(torch.bitwise_and(in_int, nan_mask) == 0x7f800000, False, True)
 
    # perform round nearest tie even
    sr = torch.tensor(13, device=device, dtype=torch.int32)
    one = torch.tensor(1, device=device, dtype=torch.int32)
    point5 = torch.tensor(0x00000fff, device=device, dtype=torch.int32)
    fixup = torch.bitwise_and(torch.bitwise_right_shift(in_int, sr), one)
    in_round = in_int + point5 + fixup
    in_round = torch.where(do_round, in_round, in_int)
 
    output = torch.bitwise_and(in_round, mask).view(torch.float)
 
    return output


@torch.no_grad()
def cast_fp32_tf32_float(input, mode="rne"):  # type(input) != tuple
    input = torch.tensor(input)
    output = cast_fp32_tf32(input, mode=mode)
    output = float(output)
    return output

# @torch.no_grad()
# def cast_fp32_tf32_tuple(input_tuple, mode="rne"):  # type(input)=tuple
#     assert type(input_tuple) == tuple
#     output_list=[]
#     for input in input_tuple: 
#         # print('input:', input.grad, input)
#         output = cast_fp32_tf32(input, mode=mode)
#         # print('output0:', output.grad, output)
#         # output=output.requires_grad_(True)
#         # print('output1:', output.grad, output)
#         output_list.append(output)

#     return tuple(output_list) 


@torch.no_grad()
def cast_fp32_tf32_tuple(input_tuple, mode="rne"):  # type(input)=tuple
    assert type(input_tuple) == tuple
    # print('Before cast:', len(input_tuple), input_tuple[0].shape, input_tuple[0][0][0][0])
    # starttime = time.time()
    for i,input in enumerate(input_tuple): 
        if(input!=None):
            cast_fp32_tf32_inplaceV2(input, mode=mode)  # modify inplace, thus no return 
        # else:
        #     print('CAST: input is None. index=',i)  # need to further verify
        # cast_fp32_tf32_storage(input.cpu(), mode=mode)  # modify inplace, thus no return 
        # print('After cast0:', input)
    # endurance = time.time()-starttime
    # print('endurance:',endurance)
    # print('After cast1:', len(input_tuple), input_tuple[0].shape, input_tuple[0][0][0][0])
    # input('stop here')
    # return input_tuple

###############################################################################
layer_type=(nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, SelectAdaptivePool2d, nn.Linear)
# def forward_pre_hook(module, inputdata):
#     inputdata = cast_fp32_tf32_tuple(inputdata)
#     return inputdata

def forward_pre_hook(module, inputdata):
    cast_fp32_tf32_tuple(inputdata)

def forward_pre_hook_verbose(module, inputdata): 
    # print('HOOK before cast:', type(inputdata),len(inputdata), inputdata[0].shape, inputdata[0][0][0][0].shape)  #HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 3, 160, 160]) torch.Size([160])
    # print('HOOK before cast:', inputdata[0][0][0][0])
    print('forward_pre_hook before cast:', type(inputdata),len(inputdata),inputdata[0].shape)
    print('forward_pre_hook before cast:', inputdata[0][0][0][0])
    # inputdata = cast_fp32_tf32_tuple(inputdata)
    cast_fp32_tf32_tuple(inputdata)
    print('forward_pre_hook after cast:', type(inputdata),len(inputdata),inputdata[0].shape)
    print('forward_pre_hook after cast:', inputdata[0][0][0][0])
    input('stop here')
    # print('HOOK after cast:', type(inputdata),len(inputdata), inputdata[0].shape, inputdata[0][0][0][0].shape)  #HOOK OUT: <class 'tuple'> 1 torch.Size([256, 3, 160, 160])
    # print('HOOK after cast:', inputdata[0][0][0][0])
    # return inputdata

def forward_hook(module, inputdata, outputdata):
    # print('forward_hook before cast:', type(inputdata),len(inputdata),inputdata[0].shape) # HOOK before cast: <class 'tuple'> 2 torch.Size([128, 1000])
    # print('forward_hook before cast:', inputdata[0][0][0][0])
    # input('---stop here---')
    # print('HOOK before cast:', type(outputdata),outputdata) # HOOK before cast: <class 'torch.Tensor'> tensor(0.7046, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
    # outputdata = cast_fp32_tf32(outputdata).requires_grad_(True)
    cast_fp32_tf32_inplaceV2(outputdata) #,mode='trunc'
    # print('HOOK after cast:', type(outputdata),outputdata) # HOOK after cast: <class 'torch.Tensor'> tensor(0.7046, device='cuda:0')
    # input('stop here')
    # return outputdata


def backward_hook(module, grad_input, grad_output):
    # print(module)
    # if (grad_input is None) or (grad_output is None):
    #     print('Meet NoneType.')
    # else:
    #     print('    input:', len(grad_input), grad_input[0].shape)
    #     print('    output:', len(grad_output), grad_output[0].shape)
    # print('HOOK before cast:', type(grad_input),len(grad_input), grad_input[0].shape)  #HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 3, 160, 160]) torch.Size([160])
    # print('before grad_input:', grad_input[0][0])
    # grad_input = cast_fp32_tf32_tuple(grad_input)
    cast_fp32_tf32_tuple(grad_input)
    # print('HOOK after cast:', type(grad_input),len(grad_input), grad_input[0].shape)  #HOOK OUT: <class 'tuple'> 1 torch.Size([256, 3, 160, 160])
    # print('after grad_input:', grad_input[0][0])
    # input('stop here')
    # return grad_input 


def register_hook(grad):
    # print('    grad:', len(grad), grad.shape)
    # print('before', grad)
    grad = cast_fp32_tf32(grad)
    # print('    grad:', len(grad), grad.shape)
    # print('after', grad)
    # input('---stop here---')
    return grad #.requires_grad_(True)


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
        print("-",name,m)
    # for idx,(name,m) in enumerate(module._modules.items()):
    #     print(idx,"-",name,m)
    input('Here to stop!')

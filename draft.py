from re import A
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
   
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
   
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def hook(module, input, output):
    '''把这层的输出拷贝到features中'''
    print('input:', input.shape)
    print('output:', output.shape)

# def backward_hook(module, grad_input, grad_output):
#     print('input:', grad_input.shape)
#     print('output:', grad_output.shape)


def register_hook_weight(grad):
    # print('    grad:', len(grad), grad.shape)
    print('before', grad)
    grad = cast_fp32_tf32(grad)
    print('after', grad)
    return grad

def backward_hook(module, grad_input, grad_output):
    # print('HOOK before cast:', type(grad_input),len(grad_input), grad_input[0].shape) # fc  HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 2048])
    # print(grad_input[0][0])
    print('HOOK before cast:', type(grad_input),len(grad_input), grad_input[0].shape, grad_input[0][0][0][0].shape)  #HOOK OUT before cast: <class 'tuple'> 1 torch.Size([256, 3, 160, 160]) torch.Size([160])
    print('HOOK before cast:', grad_input[0][0][0][0])
    grad_input = cast_fp32_tf32_tuple(grad_input)
    print('HOOK after cast:', type(grad_input),len(grad_input), grad_input[0].shape, grad_input[0][0][0][0].shape)  #HOOK OUT: <class 'tuple'> 1 torch.Size([256, 3, 160, 160])
    print('HOOK after cast:', grad_input[0][0][0][0])
    # print('HOOK after cast:', grad_input[0][0])
    return grad_input

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
    # input.cuda()

        # return output

@torch.no_grad()
def cast_fp32_tf32_inplace(input, mode="rne"):  # type(input) != tuple
    in_int = input.view(torch.int32)
    mask = torch.tensor(0xffffe000, dtype=torch.int32)

    in_size = input.size()
    in_storage = input.storage()
    output = torch.zeros(in_size, dtype=torch.float)
    print(output.shape)
    output.set_(source=in_storage) #become one demension
    print('input address: ', input.data_ptr())
    print('output address0: ', output.data_ptr(), output.shape, input.size())
 
    if mode == "trunc":
        # output = torch.bitwise_and(in_int, mask).view(torch.float)
        in_int = in_int.view(-1)
        print('len(in_int)', len(in_int))
        for index in range(len(in_int)):
            # print('before:',output.storage()[index])
            output.storage()[index] = float(torch.bitwise_and(in_int[index], mask).view(torch.float))
            # print('after:',output.storage()[index])
        print('output address1: ', output.data_ptr())
        print('input after cast: ',input.shape,input[0][0][0][0])
        print('output after cast: ',output.shape,output[-160:])

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
    output = torch.zeros(in_size, dtype=torch.float)
    print(output.shape)
    output.set_(source=in_storage) #become one demension
    in_int = output.view(torch.int32).view(in_size)
    print('input address: ', input.data_ptr())
    print('output address0: ', output.data_ptr(), output.shape)
    print('in_int address: ', in_int.data_ptr(), in_int.shape)
 
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
def cast_fp32_tf32_tuple(input_tuple, mode="rne"):  # type(input)=tuple
    assert type(input_tuple) == tuple
    print('Before cast:', len(input_tuple), input_tuple[0].shape, input_tuple[0][0][0][0])
    starttime = time.time()
    for input in input_tuple: 
        cast_fp32_tf32_inplaceV2(input, mode='trunc')  # modify inplace, thus no return 
        # cast_fp32_tf32_storage(input, mode=mode)  # modify inplace, thus no return .cpu()
    endurance = time.time()-starttime
    print('endurance:',endurance)
    print('After cast:', len(input_tuple), input_tuple[0].shape, input_tuple[0][0][0][0])
    # input('stop here')
    # return input_tuple

def forward_pre_hook(module, inputdata):
    cast_fp32_tf32_tuple(inputdata)

# net = LeNet()
# handle = net.register_forward_pre_hook(forward_pre_hook)
# # handle0 = net.conv1.register_full_backward_hook(hook)
# # handle1 = net.conv2.register_full_backward_hook(hook)
# # handle2 = net.fc1.register_full_backward_hook(hook)
# # handle3 = net.fc2.register_full_backward_hook(hook)
# # handle4 = net.fc3.register_full_backward_hook(hook)
# # img = t.rand(1,1,32,32)
# img = t.rand(128,3,160,160) #.to(device='cuda')
# print(img.shape, img)
# output = net(img)
# # output.backward()
# # 用完hook后删除
# handle.remove()
# # handle0.remove()
# # handle1.remove()
# # handle2.remove()
# # handle3.remove()
# # handle4.remove()


# a = torch.randn(8)
# print('a:',a)
# v = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], requires_grad=True, dtype=torch.float32)
# # h = v.register_hook(lambda grad: grad * 2)  # double the gradient
# h = v.register_hook(register_hook_weight)  # double the gradient

# v.backward(a)
# # 先计算原始梯度，再进hook，获得一个新梯度。
# print(v.grad.data)
# h.remove()  # removes the hook



## below is unit test
dtype = torch.float
device = torch.device("cpu")
# a = torch.randn(8, device=device, dtype=dtype)
tmp=[0.7051,0.7041,0.7046,0.7034,0.7022,0.7017]
a = torch.tensor(tmp, device=device, dtype=dtype)
print(a)
# b = cast_fp32_tf32_gpu(a, mode="rne") #1063M
cast_fp32_tf32_inplaceV2(a, mode="trunc") #1063M
# b_inplace = cast_fp32_tf32_inplace(a, mode="rne") #1063M
print("trunc")
print(a, type(a), a.grad)
# print(b_inplace, type(b_inplace), b_inplace.grad)
 
# c = cast_fp32_tf32(a, mode="trunc")
# print("trunc")
# print(c, c.grad)
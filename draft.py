import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
   
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
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

def backward_hook(module, grad_input, grad_output):
    print('input:', grad_input.shape)
    print('output:', grad_output.shape)
 
# net = LeNet()
# # handle = net.register_forward_hook(hook)
# handle0 = net.conv1.register_full_backward_hook(hook)
# handle1 = net.conv2.register_full_backward_hook(hook)
# handle2 = net.fc1.register_full_backward_hook(hook)
# handle3 = net.fc2.register_full_backward_hook(hook)
# handle4 = net.fc3.register_full_backward_hook(hook)
# img = t.rand(1,1,32,32)
# print(img.shape, img)
# output = net(img)
# output.backward()
# # 用完hook后删除
# # handle.remove()
# handle0.remove()
# handle1.remove()
# handle2.remove()
# handle3.remove()
# handle4.remove()

a = torch.randn(4)
print('a:',a)
v = torch.tensor([0, 0, 0, 0], requires_grad=True, dtype=torch.float32)
# h = v.register_hook(lambda grad: grad * 2)  # double the gradient
h = v.register_hook(register_hook_weight)  # double the gradient

v.backward(a)
# 先计算原始梯度，再进hook，获得一个新梯度。
print(v.grad.data)
h.remove()  # removes the hook



def register_hook_weight(grad):
    print('    grad:', len(grad), grad.shape)
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
def cast_fp32_tf32_tuple(input_tuple, mode="rne"):  # type(input)=tuple
    assert type(input_tuple) == tuple

    output_list=[]
    for input in input_tuple: 
        # print('input:',input[0][0][0][0])
        in_int = input.view(torch.int32)
        # print('in_int:',in_int[0][0][0])
        mask = torch.tensor(0xffffe000, dtype=torch.int32)
    
        if mode == "trunc":
            output = torch.bitwise_and(in_int, mask).view(torch.float)
            return (output,)
    
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
        # print('output:',output[0][0][0])
        output_list.append(output)
 
    return tuple(output_list)
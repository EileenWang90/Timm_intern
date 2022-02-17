import torch
import torch

@torch.no_grad()
def cast_fp32_tf32_inplaceV2(input, mode="rne"):  # type(input) != tuple
    mask = torch.tensor(0xffffe000, dtype=torch.int32)

    in_size = input.size()
    in_storage = input.storage()
    output = torch.zeros(in_size, dtype=torch.float, device=input.device)
    output.set_(source=in_storage) #become one demension; share same storage with input
    in_int = output.view(torch.int32).view(in_size)

    if mode == "trunc":
        in_int.bitwise_and_(mask)
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
        # in_round = torch.where(do_round, in_round, in_int) # can use inplace mask-mul-add to implement

        in_int.bitwise_and_(mask)

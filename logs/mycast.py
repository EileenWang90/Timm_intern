import torch
import torch.nn as nn

 ##############################################################################################
 ## the first version of Forward Pre Hook  in train.py line390
 ## using named_children(), and specify submodules to prehook
    ### Forward Pre Hook
    from timm.models.layers import SelectAdaptivePool2d
    handle=[]
    layer_type=(nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, SelectAdaptivePool2d, nn.Linear)
    for index, (name, child) in enumerate(model.named_children()):
        # print(index, name, child)
        if isinstance(child, layer_type):  # manual
            print('###', index, name, type(child), child)
            tmpstr = 'model' + '.'+ name + '.register_forward_pre_hook(forward_pre_hook)'
            print(tmpstr)
            handle.append(eval(tmpstr))
        elif isinstance(child, nn.Sequential):
            assert name in ('layer1','layer2','layer3','layer4')
            bottleneckdict={'layer1':3,'layer2':4,'layer3':6,'layer4':3}
            print('### Sequential',index, name, type(child))#, child)
            for i in range(bottleneckdict[name]):
                layername='model.'+name+'['+str(i)+']'
                # print(layername)
                for index, (subname, child) in enumerate(eval(layername+'.named_children()')):
                    # print('---', index, subname, type(child), child)
                    if isinstance(child, layer_type):  # manual
                        tmpstr = layername + '.'+ subname + '.register_forward_pre_hook(forward_pre_hook)'
                        print(tmpstr)  # model.layer1[0].conv1.register_forward_pre_hook(forward_pre_hook)
                        handle.append(eval(tmpstr))
                    elif isinstance(child, nn.Sequential) and (subname == 'downsample'):
                        assert subname == 'downsample'
                        tmpstr0 = layername + '.'+ subname + '[0].register_forward_pre_hook(forward_pre_hook)'
                        tmpstr1 = layername + '.'+ subname + '[1].register_forward_pre_hook(forward_pre_hook)'
                        print(tmpstr0)
                        print(tmpstr1)
                        handle.append(eval(tmpstr0))
                        handle.append(eval(tmpstr1))
                        break
                    else:
                        _logger.error('Meet unknown layer:', subname, 'when dealing with submodule forward_pre_hook')                
        else:
            _logger.error('Meet unknown layer:', name, 'when dealing with forward_pre_hook')
            # print('ooo', index, name, type(child), child)

    ### Usage
    # handle = model.conv1.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.bn1.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.act1.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.maxpool.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.layer1[0].conv1.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.layer1[0].downsample[0].register_forward_pre_hook(assistant.forward_pre_hook) #hook  
    # handle = model.layer1[0].downsample[1].register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.global_pool.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.fc.register_forward_pre_hook(assistant.forward_pre_hook) #hook
    # handle = model.register_forward_pre_hook(assistant.print_hook) #hook
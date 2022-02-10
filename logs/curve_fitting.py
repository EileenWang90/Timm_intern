import matplotlib.pyplot as plt

filelist = ['20220118ResNetV1.5FP32','20220119ResNetV1.5TF32','20220209ResNetV1.5AMP']
filepath = './interlog/'
# filename = filelist[0] 

#########################################################################
## file preprocess
for filename in filelist:
    f_in = open(filename+'.log',"r+")
    f_train_out = open(filepath+filename+'_train.log',"w+")
    f_eval_out = open(filepath+filename+'_eval.log',"w+")
    count=0
    for readline in f_in.readlines():
        readline = readline[61:] #get rid of training time
        # num = readline.rfind("Train: ") #61
        # print(num, readline)
        # if(num != -1):
        #     input('here to stop')
        
        # if readline.startswith('Train: ') or readline.startswith('Test: '):
        if (readline.startswith('Train: ') and readline.find('100%')!= -1): 
            count += 1
            # print(readline.find('100%'))
            f_train_out.write(readline)
        if readline.startswith('Test: [  24/24]  '):
            f_eval_out.write(readline)
        # if count==20:
        #     break
    f_in.close()
    f_train_out.close()
    f_eval_out.close()

#########################################################################
### get loss list
train_loss_whole=[]
eval_loss_whole=[]
eval_top1_whole=[]
eval_top5_whole=[]
for filename in filelist:
    train_loss=[]
    eval_loss=[]
    eval_top1=[]
    eval_top5=[]
    f_train_out = open(filepath+filename+'_train.log',"r")
    f_eval_out = open(filepath+filename+'_eval.log',"r")
    index = 0
    for train_line,eval_line in zip(f_train_out.readlines(),f_eval_out.readlines()):
        index += 1
        trainloss_tmp = train_line.strip('\n').split("  ")[1]
        trainloss_ave = float(trainloss_tmp.split(' ')[2][1:-1])
        train_loss.append(trainloss_ave)

        evalloss_tmp = eval_line.strip('\n').split(":")
        evalloss_start, evalloss_end = evalloss_tmp[3].find('('), evalloss_tmp[3].find(')')
        evaltop1_start, evaltop1_end = evalloss_tmp[4].find('('), evalloss_tmp[3].find(')')
        evaltop5_start, evaltop5_end = evalloss_tmp[5].find('('), evalloss_tmp[3].find(')')
        evalloss_ave = float(evalloss_tmp[3][evalloss_start+1:evalloss_end])
        evaltop1_ave = float(evalloss_tmp[4][evaltop1_start+1:evaltop1_end].strip(' '))
        evaltop5_ave = float(evalloss_tmp[5][evaltop5_start+1:evaltop5_end].strip(' '))
        eval_loss.append(evalloss_ave)
        eval_top1.append(evaltop1_ave)
        eval_top5.append(evaltop5_ave)
        # if index < 20:
            # print(evalloss_tmp, evalloss_ave)
            # print(evalloss_ave, evaltop1_ave, evaltop5_ave)
            # print(evalloss_tmp, type(evalloss_ave), evalloss_ave)
    f_train_out.close()
    f_eval_out.close()
    train_loss_whole.append(train_loss)
    eval_loss_whole.append(eval_loss)
    eval_top1_whole.append(eval_top1)
    eval_top5_whole.append(eval_top5)
    # print('train_loss:', train_loss, len(train_loss))
    # print('eval_loss:', eval_loss, len(eval_loss))
    # print('eval_top1:', eval_top1, len(eval_top1))
    # print('eval_top5:', eval_top5, len(eval_top5), '\n')

##########################################################################
### draw curve
epoch=range(0,100)
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)

ax1.plot(epoch,train_loss_whole[0],label='FP32')
ax1.plot(epoch,train_loss_whole[1],label='TF32')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss')
ax1.legend()

ax2.plot(epoch,eval_loss_whole[0],label='FP32')
ax2.plot(epoch,eval_loss_whole[1],label='TF32')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Eval Loss')
ax2.set_title('Eval Loss')
ax2.legend()

ax3.plot(epoch,eval_top1_whole[0],label='FP32')
ax3.plot(epoch,eval_top1_whole[1],label='TF32')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Eval Top1')
ax3.set_title('Eval Top1')
ax3.legend()

ax4.plot(epoch,eval_top5_whole[0],label='FP32')
ax4.plot(epoch,eval_top5_whole[1],label='TF32')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Eval Top5')
ax4.set_title('Eval Top5')
ax4.legend()

plt.show()
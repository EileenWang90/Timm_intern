import matplotlib.pyplot as plt

# filelist = ['20220118ResNetV1.5FP32','20220119ResNetV1.5TF32','20220209ResNetV1.5AMP']
filelist = ['summary_FP32.csv','summary_TF32.csv','summary_AMP.csv','../2020216ResNetV1.5wholeTF32v0.4.log']
# filepath = './interlog/'
# filename = filelist[0] 


##########################################################################
### get loss list
train_loss_whole=[]
eval_loss_whole=[]
eval_top1_whole=[]
eval_top5_whole=[]
for k, filename in enumerate(filelist):
    train_loss=[]
    eval_loss=[]
    eval_top1=[]
    eval_top5=[]
    f_summary = open(filename,"r")
    index = 0
    for i, readline in enumerate(f_summary.readlines()):
        if i != 0: # rid of line0
            readline = readline.strip('\n')
            # if i < 20:
            #     print(readline)
            tmplist = readline.split(',')
            train_loss.append(float(tmplist[1]))
            eval_loss.append(float(tmplist[2]))
            eval_top1.append(float(tmplist[3]))
            eval_top5.append(float(tmplist[4]))
    f_summary.close()
    train_loss_whole.append(train_loss)
    eval_loss_whole.append(eval_loss)
    eval_top1_whole.append(eval_top1)
    eval_top5_whole.append(eval_top5)
    # print('\n', k)
    # print('train_loss:', train_loss, len(train_loss), '\n')
    # print('eval_loss:', eval_loss, len(eval_loss), '\n')
    # print('eval_top1:', eval_top1, len(eval_top1), '\n')
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
ax1.plot(epoch,train_loss_whole[2],label='AMP')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss')
ax1.legend()

ax2.plot(epoch,eval_loss_whole[0],label='FP32')
ax2.plot(epoch,eval_loss_whole[1],label='TF32')
ax2.plot(epoch,eval_loss_whole[2],label='AMP')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Eval Loss')
ax2.set_title('Eval Loss')
ax2.legend()

ax3.plot(epoch,eval_top1_whole[0],label='FP32')
ax3.plot(epoch,eval_top1_whole[1],label='TF32')
ax3.plot(epoch,eval_top1_whole[2],label='AMP')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Eval Top1')
ax3.set_title('Eval Top1')
ax3.legend()

ax4.plot(epoch,eval_top5_whole[0],label='FP32')
ax4.plot(epoch,eval_top5_whole[1],label='TF32')
ax4.plot(epoch,eval_top5_whole[2],label='AMP')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Eval Top5')
ax4.set_title('Eval Top5')
ax4.legend()

plt.show()

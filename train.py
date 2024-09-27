import os
import argparse
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from BraTS import *
from BraTS_plus.networks.Attention_GE import AttensionUNet_GE
from BraTS_plus.networks.Unet_GE import UNet_GE
from BraTS_plus.networks.AttentionUnet import AttentionUNet
from BraTS_plus.networks.Unet import UNet
from utils import Loss,cal_dice,cosine_scheduler


def train_loop(model,optimizer,scheduler,criterion,train_loader,device,epoch):
    model.train()
    running_loss = 0
    dice1_train = 0
    dice2_train = 0
    dice3_train = 0
    pbar = tqdm(train_loader)
    for it,(images,masks) in enumerate(pbar):#遍历训练数据加载器中的每个批次，it是批次索引，images和masks分别是输入图像和对应的标签（分割掩码）。

        # update learning rate according to the schedule
        it = len(train_loader) * epoch + it
        param_group = optimizer.param_groups[0]
        param_group['lr'] = scheduler[it]
        # print(scheduler[it])

        # [b,4,128,128,128] , [b,128,128,128]
        images, masks = images.to(device),masks.to(device)          #将图像和标签移动到指定设备：images.to(device)和masks.to(device)。前向传播：计算模型的输出outputs。
        # [b,4,128,128,128], 4分割
        outputs = model(images)
        # outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, masks)
        dice1, dice2, dice3 = cal_dice(outputs,masks)
        pbar.desc = "loss: {:.3f} ".format(loss.item())
        #将当前批次的损失和评估指标累加到总和中。
        running_loss += loss.item()
        dice1_train += dice1.item()
        dice2_train += dice2.item()
        dice3_train += dice3.item()
        #反向传播和参数更新
        optimizer.zero_grad()#清零优化器的梯度。
        loss.backward()#计算损失的梯度。
        optimizer.step()#根据梯度更新模型参数。
    loss = running_loss / len(train_loader)
    dice1 = dice1_train / len(train_loader)
    dice2 = dice2_train / len(train_loader)
    dice3 = dice3_train / len(train_loader)
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3}


def val_loop(model,criterion,val_loader,device):
    model.eval()
    running_loss = 0
    dice1_val = 0
    dice2_val = 0
    dice3_val = 0
    pbar = tqdm(val_loader)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.softmax(outputs,dim=1)

            loss = criterion(outputs, masks)
            dice1, dice2, dice3 = cal_dice(outputs, masks)

            running_loss += loss.item()
            dice1_val += dice1.item()
            dice2_val += dice2.item()
            dice3_val += dice3.item()
            # pbar.desc = "loss:{:.3f} dice1:{:.3f} dice2:{:.3f} dice3:{:.3f} ".format(loss,dice1,dice2,dice3)

    loss = running_loss / len(val_loader)
    dice1 = dice1_val / len(val_loader)
    dice2 = dice2_val / len(val_loader)
    dice3 = dice3_val / len(val_loader)
    return {'loss':loss,'dice1':dice1,'dice2':dice2,'dice3':dice3}


def train(model,optimizer,scheduler,criterion,train_loader,
          val_loader,epochs,device,train_log,valid_loss_min=999.0):
    for e in range(epochs):
        # train for epoch
        train_metrics = train_loop(model,optimizer,scheduler,criterion,train_loader,device,e)
        # eval for epoch
        val_metrics = val_loop(model,criterion,val_loader,device)
        info1 = "Epoch:[{}/{}] train_loss: {:.3f} valid_loss: {:.3f} ".format(e+1,epochs,train_metrics["loss"],val_metrics["loss"])
        info2 = "Train--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(train_metrics['dice1'],train_metrics['dice2'],train_metrics['dice3'])
        info3 = "Valid--ET: {:.3f} TC: {:.3f} WT: {:.3f} ".format(val_metrics['dice1'],val_metrics['dice2'],val_metrics['dice3'])
        print(info1)
        print(info2)
        print(info3)
        with open(train_log,'a') as f:
            f.write(info1 + '\n' + info2 + ' ' + info3 + '\n')

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict()}
        if val_metrics['loss'] < valid_loss_min:
            valid_loss_min = val_metrics['loss']
            torch.save(save_file, 'E:\\DataSet\\Data\\default\\UNet.pth')
        else:
            torch.save(save_file,os.path.join(args.save_path,'checkpoint{}.pth'.format(e+1)))
    print("Finished Training!")


def main(args):
    torch.manual_seed(args.seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)  # 为所有的GPU设置种子，以使得结果是确定的

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cpu')

    # data info
    patch_size = (160,160,128)
    train_dataset = BraTS(args.data_path,args.train_txt,transform=transforms.Compose([
        RandomRotFlip(),
        RandomCrop(patch_size),
        #GaussianNoise(p=0.1),
        ToTensor()
    ]))
    val_dataset = BraTS(args.data_path,args.valid_txt,transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))
    test_dataset = BraTS(args.data_path,args.test_txt,transform=transforms.Compose([
        CenterCrop(patch_size),
        ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=12,   # num_worker=4
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                            pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=12, shuffle=False,
                             pin_memory=True)

    print("using {} device.".format(device))
    print("using {} images for training, {} images for validation.".format(len(train_dataset), len(val_dataset)))
    # img,label = train_dataset[0]

    # 1-坏疽(NT,necrotic tumor core),2-浮肿区域(ED,peritumoral edema),4-增强肿瘤区域(ET,enhancing tumor)
    # 评价指标：ET(label4),TC(label1+label4),WT(label1+label2+label4)
    '''
UNet 是一种常用于医学图像分割的卷积神经网络模型。它由编码器和解码器组成，能够有效地捕捉图像中的多尺度特征。
in_channels=4 表示输入的图像有4个通道。这通常对应于MRI的四个模态（T1、T1Gd、T2、FLAIR）。
num_classes=4 表示输出的分割结果有4个类别（包括背景和三种不同的肿瘤区域）。
.to(device) 将模型移动到指定的设备（例如CPU或GPU）上，以便进行计算。

Loss 是自定义的损失函数类，通常是基于交叉熵损失或Dice损失等。
n_classes=4 表示分类任务中有4个类别。
weight=torch.tensor([0.2, 0.3, 0.25, 0.25]) 设置各类别的权重，帮助模型更好地平衡不同类别之间的训练。
.to(device) 将损失函数的计算也移动到指定设备上，以便与模型计算一致。

optim.SGD 是随机梯度下降优化器（Stochastic Gradient Descent）。
model.parameters() 将模型的参数传递给优化器，以便在训练过程中更新这些参数。
momentum=0.9 添加动量项，可以加速收敛和减小震荡。
lr=0 初始学习率设为0，实际的学习率会由学习率调度器控制。
weight_decay=5e-4 权重衰减（L2正则化）项，有助于防止过拟合。

cosine_scheduler 是一个自定义的学习率调度器函数，基于余弦退火策略调整学习率。
base_value=args.lr 表示初始学习率。
final_value=args.min_lr 表示训练结束时的最小学习率。
epochs=args.epochs 表示训练的总轮数。
niter_per_ep=len(train_loader) 表示每个epoch的迭代次数（训练集数据加载器的长度）。
warmup_epochs=args.warmup_epochs 表示学习率预热的轮数，在这段时间内学习率从start_warmup_value逐渐增加到base_value。
start_warmup_value=5e-4 表示预热开始时的学习率。
    '''
    model = AttentionUNet(in_channels=4,num_classes=4).to(device)
    criterion = Loss(n_classes=4, weight=torch.tensor([0.2, 0.3, 0.25, 0.25])).to(device)
    optimizer = optim.SGD(model.parameters(),momentum=0.9, lr=0, weight_decay=5e-4)
    scheduler = cosine_scheduler(base_value=args.lr,final_value=args.min_lr,epochs=args.epochs,
                                 niter_per_ep=len(train_loader),warmup_epochs=args.warmup_epochs,start_warmup_value=5e-4)

    # 加载训练模型
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optimizer'])
        print('Successfully loading checkpoint.')

    train(model,optimizer,scheduler,criterion,train_loader,val_loader,args.epochs,device,train_log=args.train_log)

    # metrics1 = val_loop(model, criterion, train_loader, device)
    metrics2 = val_loop(model, criterion, val_loader, device)
    metrics3 = val_loop(model, criterion, test_loader, device)

    # 保存模型
    torch.save(model.state_dict(), 'E:\\DataSet\\Data\\default\\model.pth')

    # 最后再评价一遍所有数据，注意，这里使用的是训练结束的模型参数
    # print("Train -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics1['loss'], metrics1['dice1'],metrics1['dice2'], metrics1['dice3']))
    print("Valid -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics2['loss'], metrics2['dice1'], metrics2['dice2'], metrics2['dice3']))
    print("Test  -- loss: {:.3f} ET: {:.3f} TC: {:.3f} WT: {:.3f}".format(metrics3['loss'], metrics3['dice1'], metrics3['dice2'], metrics3['dice3']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--epochs', type=int, default=15)#每轮中训练集一次valid一次，所有轮次结束后valid和test各一次
    parser.add_argument('--warmup_epochs', type=int, default=0)#预热轮次
    parser.add_argument('--batch_size', type=int, default=1)#每次从训练集中选取的images数量，增大batchsize训练速度并不会有明显提高，但准确率有提高，超过2会给CPU干烧了
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--min_lr', type=float, default=0.002)
    parser.add_argument('--data_path', type=str, default='E:\\DataSet\\Data\\output')
    parser.add_argument('--train_txt', type=str, default='E:\\DataSet\\Data\\output\\train.txt')
    parser.add_argument('--valid_txt', type=str, default='E:\\DataSet\\Data\\output\\valid.txt')
    parser.add_argument('--test_txt', type=str, default='E:\\DataSet\\Data\\output\\test.txt')
    parser.add_argument('--train_log', type=str, default='E:\\DataSet\\Data\\default\\UNet.txt')
    parser.add_argument('--weights', type=str, default='E:\\DataSet\\Data\\default\\UNet.pth')
    parser.add_argument('--save_path', type=str, default='E:\\DataSet\\Data\\default')

    args = parser.parse_args()

    main(args)

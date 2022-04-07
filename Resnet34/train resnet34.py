import numpy as np
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch
import os
from my_transform_3 import transform
from my_image_folder import ImageFolder
from network import resnet34
def testset_loss(dataset, network):

    loader = torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=2)

    all_loss = 0.0
    for i,data in enumerate(loader,0):

        inputs,labels = data
        inputs = Variable(inputs)

        outputs = network(inputs)
        all_loss = all_loss + abs(labels[0]-outputs.data[0][0])

    return all_loss/i



def IndexNotensor(f):
    all_mae = 0
    all_mse = 0
    bias = 0
    count = 0
    wind_real_list = []
    wind_pre_list = []
    for line in f:
        line1 = line.split(',')[1]
        line2 = line1.split("(")[1]
        wind_pre = float(line2)
        wind_real = line.split('_')[2]
        wind_real = float(wind_real)
        bias = bias + (wind_pre - wind_real)
        all_mae = all_mae + abs(wind_pre - wind_real)
        all_mse = all_mse + (wind_pre - wind_real) ** 2
        wind_real_list.append(wind_real)
        wind_pre_list.append(wind_pre)
        count = count + 1
    f.close()
    MAE = all_mae / count
    Bias = bias / count
    RMSE = all_mse / count
    RMSE = np.sqrt(RMSE)
    return MAE, RMSE, Bias



if __name__ == '__main__':
    imgsize = 256
    path_ = os.path.abspath(r'/content/drive/MyDrive/TCIE/train_set')
    trainset = ImageFolder(path_, imgsize, transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=16, shuffle=True, num_workers=1)
    print(trainloader)
    testset = ImageFolder(r'/content/drive/MyDrive/TCIE/test_set',imgsize, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=1)
    pathoutput = r"/content/drive/MyDrive/TCIE/resnet"
    pathlosssave = os.path.join(r'/content/drive/MyDrive/TCIE/resnet')
    tys_time = {} 
    totalloss = []
    test_allloss = []
    max_RMSE = 0
    if not os.path.exists(pathlosssave):
        os.makedirs(pathlosssave)
    if not os.path.exists(pathoutput):
        os.makedirs(pathoutput)
    model_path = r"/content/drive/MyDrive/TCIE/resnet"
    net = resnet34(pretrained=False, modelpath=model_path, num_classes=1000)
    net.fc = nn.Sequential(nn.Linear(2048, 512),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(512, 64),
                           nn.ReLU(),
                           nn.Linear(64, 1))
    net.cuda()
    print(net)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    for epoch in range(15):

        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
          
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs = net(inputs)
            new_shape = (len(labels), 1)
            labels = labels.view(new_shape)
            loss = criterion(outputs,labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch+1,i+1,running_loss/100))
                totalloss.append(running_loss / 100)
                running_loss = 0.0
        if 5 < epoch < 50:
            if epoch % 2 == 1 :
                torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net.pth')
        else:
            if epoch % 20 == 19 :
                torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net.pth')

        net.eval()
        all_loss = 0.0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            new_shape = (len(labels), 1)
            labels = labels.view(new_shape)
            test_loss = criterion(outputs, labels.float())
            all_loss += test_loss.data
            wind = outputs.data[0][0]
            name = testset.__getitemName__(i)
            name = name.split('_')
            tid_time = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[3]
            tys_time[tid_time] = wind

        f = open(pathlosssave + r'/result_' + str(epoch + 1) + '.txt', 'a')
        f2 = open(pathlosssave + r'/resnet test_epoch_loss.txt', 'a')

        test_allloss.append((all_loss / i))
        tys_time = sorted(tys_time.items(), key=lambda asd: asd[0], reverse=False)
        for ty in tys_time:
            f.write(str(ty) + '\n')
        f.close()
        f_r = open(pathlosssave + r'/result_' + str(epoch + 1) + '.txt')
        MAE, RMSE, Bias = IndexNotensor(f_r)
        line = 'epoch %d, testloss: %4f MAE: %4f RMSE: %4f Bias: %4f\n' % (epoch + 1, (all_loss / i), MAE, RMSE, Bias)
        print(line)
        if RMSE < max_RMSE:
            max_RMSE = RMSE
            torch.save(net.state_dict(), pathoutput + '/' + str(epoch + 1) + '_net.pth')
        tys_time = {}
        f2.write(line)
        net.train()
    print('Finished Training')
    torch.save(net.state_dict(), pathoutput+'/resnet_relu.pth')
    np.savetxt(pathlosssave + r'\total_loss.csv'
               , totalloss, delimiter=',')
    
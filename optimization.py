import math
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data import Data_util
from transformer import Transformer
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torch.utils.tensorboard import SummaryWriter

L1 = nn.L1Loss(reduction='sum')
L2 = nn.MSELoss(reduction='sum')


# writer=SummaryWriter('./logs')
def train(model, optimizer, Data, X, Y, epoch, device, criterion, batch_size):
    model.train()
    batch_id = 0
    total_loss = 0
    n_samples = 0
    for data, target in Data.get_batches(X, Y, batch_size):  # 通过迭代器data.get_batches（）往外给处每批训练数据
        data, target = data.to(device), target.to(device)
        #print('-----',target.shape)
        # to(device) 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上，之后的运算都在GPU上进行，
        optimizer.zero_grad()  # 将模型的参数梯度初始化为0，这个是每一轮的batch都要清除一次梯度。
        output = model(data, target)  # 前向传播计算预测值
        loss = criterion(output, target)  # 计算损失函数
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新所有参数
        total_loss += loss.item()  # 取出张量具体位置的元素值，类似于value。这样做是为了节省内存
        n_samples += (output.size(0) * Data.output_dim)
        if (batch_id + 1) % 30 == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
        batch_id += 1
    return total_loss / n_samples


def evaluate(model, Data, X, Y, device, batch_size):
    model.eval()
    total_loss_l1 = 0
    total_loss_l2 = 0
    n_samples = 0
    predict = None
    test = None

    for data, target in Data.get_batches(X, Y, batch_size):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():  # 当进行神经网络的参数更新的过程中, 我们不希望将有些的参数进行梯度下降进行更新,
            output = model(data,target)
        if predict is None:
            predict = output
            test = target
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, target))

        total_loss_l1 += L1(output, target).item()  # item()作用是取出单元张量的元素值并返回该值。
        total_loss_l2 += L2(output, target).item()
        n_samples += (output.size(0) * Data.output_dim)

    loss = total_loss_l2 / n_samples

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = predict.std(axis=0)
    sigma_g = Ytest.std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)

    rse = math.sqrt(total_loss_l2) / math.sqrt(np.square(predict - mean_p).sum())

    index = (sigma_g != 0)
    #correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    #correlation = (correlation[index]).mean()
    return loss, rse#, correlation


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--window_len', type=int, default=10,
                    help='seq_len')
parser.add_argument('--target_len', type=int, default=1)
parser.add_argument('--num_encoder_layers', type=int, default=8)
parser.add_argument('--num_decoder_layers', type=int, default=8)
parser.add_argument('--input_size', type=int, default=57,
                    help='The size of input')
parser.add_argument('--out_size', type=int, default=41,
                    help='The size of output_size')
parser.add_argument('--d_model', type=int, default=512,
                    help='The size of output')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=8, metavar='N')
parser.add_argument('--feedforward', type=int, default=64, metavar='N')
parser.add_argument('--save', type=str, default='model_skip.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--positional_encoding', type=str, default='sinusoidal')
parser.add_argument('--L1Loss', type=bool, default=False)
args = parser.parse_args()

Data = Data_util(0.6, 0.2, args.window_len, args.horizon, args.out_size, args.cuda)
print(Data.train[1].shape)
if args.L1Loss:  # 损失函数求解方法
    criterion = nn.L1Loss(reduction='sum')
else:
    criterion = nn.MSELoss(reduction='sum')

model = Transformer(args, Data)
if args.cuda:
    model.cuda()
opt = optim.Adam(model.parameters())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
best_val = 10000000

# 记录训练集和测试集的loss
train_l = []
test_l = []
star_time = time.time()
for epoch in range(args.epochs + 1):
    epoch_start_time = time.time()  # 统计时间
    # 正式开始训练，并返回损失值
    train_loss = train(model, opt, Data, Data.train[0], Data.train[1], epoch, device, criterion, args.batch_size)
    val_loss, val_rse = evaluate(model, Data, Data.valid[0], Data.valid[1], device, args.batch_size)
    epoch_end_time = time.time()
    print(
        '| end of epoch {:3d}  | time: {:5.3f}s |train_loss {:5.6f} | valid loss {:5.6f}|valid rse {:5.6f} '.format(
            epoch, (epoch_end_time - epoch_start_time), train_loss, val_loss, val_rse))
    # Save the model if the validation loss is the best we've seen so far.
    # 计算每100epoch的训练时间
    if epoch % 100 == 0:
        start_time = star_time
        end_time = time.time()
        total_time = end_time - start_time
        star_time = end_time
        print("end of 100 epoch train time: {:5.3f}".format(total_time))
    # 保存最佳模型
    # 记录训练集和测试集的loss
    train_l.append(train_loss)
    test_l.append(val_loss)

    # 保存最佳模型
    if val_loss < best_val:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        best_val = val_loss
    # 每5轮进行一次测试，计算测试集的准确率等
    if epoch % 5 == 0:
        test_loss, test_acc = evaluate(model, Data, Data.test[0], Data.test[1], device, args.batch_size)
        print("test loss {:5.6f}|test rse {:5.6f} |".format(test_loss, test_acc))

with open(args.save, 'rb') as f:
    model = torch.load(f)
test_loss, test_acc = evaluate(model, Data, Data.test[0], Data.test[1], device, args.batch_size)
print("test loss {:5.6f}|test rse {:5.6f} }".format(test_loss, test_acc))

y_pred = model(Data.test[0])
print(y_pred.shape)
y_pred_pd = pd.DataFrame(y_pred)
y_pred_pd.to_csv('pred.csv')

#  绘制训练 & 验证的损失值


# 绘制训练 & 验证的损失值
plt.plot(train_l)
plt.plot(test_l)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 取消科学技术发
np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# 求解评价指标
# 定义RMSE
def rmse(y_t, y_p):
    return np.sqrt(mean_squared_error(y_t, y_p))


def predict(model, Data, X, Y):
    model.eval()
    torch.no_grad()
    # 预测值
    y_pred = model(X)
    y_pred = y_pred.detach().numpy()
    y_pred_pd = pd.DataFrame(y_pred)
    # 真实值
    y_pred_pd.to_csv('y_pred.csv')
    y_ture = Y
    y_ture = y_ture.detach().numpy()
    y_ture_pd = pd.DataFrame(y_ture)
    y_ture_pd.to_csv('y_ture.csv')
    metrics_score = np.zeros((y_pred_pd.shape[1], 4))
    model_metrics_name = [rmse, mean_absolute_error, mean_absolute_percentage_error, r2_score]
    for i in range(y_pred_pd.shape[1]):
        for j in range(4):
            metrics_score[i, j] = model_metrics_name[j](y_ture_pd.iloc[:, i], y_pred_pd.iloc[:, i])
    metrics_score = pd.DataFrame(metrics_score, columns=['RMSE', 'MAE', 'MAPE', 'R2'])
    # print(metrics_score)
    return metrics_score


score = predict(model, Data, Data.test[0], Data.test[1])
print(score)
score.to_csv('score.csv')

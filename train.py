import torch
from torch import nn
import torch.nn.functional as F
from config import Config
from load_data import make_dataset, make_dataLoader
from model import ResNet

# 类别tensor
class_label = torch.tensor(Config.get_class_label()).cuda()

# 训练集和验证集
train_set = make_dataset(Config.get_train_path())
train_loader = make_dataLoader(train_set)
val_set = make_dataset(Config.get_val_path())
val_loader = make_dataLoader(val_set)

# 网络模型
net = ResNet()
net = net.cuda()

# 交叉熵损失函数
loss_fn = nn.CrossEntropyLoss().cuda()

# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=Config.get_learning_rate(), momentum=0.5)


# 分类准确率
def classification_acc():
    net.eval()
    total_acc = 0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = net(input)
            batch_acc = (output.argmax(1) == target).sum()
            total_acc = total_acc + batch_acc
    acc = total_acc / len(val_set)
    return acc


# 平均绝对误差
def mean_absolute_error():
    net.eval()
    total_abs_error = 0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = net(input)
            predict = F.softmax(output, dim=1)
            predict = predict * class_label
            predict = torch.sum(predict, dim=1)
            for j in range(len(target)):
                target[j] = class_label[target[j]]
            batch_abs_error = torch.sum(torch.abs(predict - target))
            total_abs_error = total_abs_error + batch_abs_error
    mae = total_abs_error / len(val_set)
    return mae


# 训练
def train():
    best_mae = 999999
    for i in range(Config.get_epoch()):
        # 训练阶段
        net.train()
        step = 0
        for input, target in train_loader:
            # 输入与输出
            input = input.cuda()
            target = target.cuda()
            output = net(input)
            # 损失函数
            loss = loss_fn(output, target)
            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step = step + 1
            if step % 10 == 0:
            	print("第{}个epoch，第{}个step，loss={}".format(i + 1, step, loss.item()))
        # 验证阶段
        net.eval()
        acc = classification_acc()
        mae = mean_absolute_error()
        print("第{}个epoch，分类准群率={},平均绝对误差={}".format(i + 1, acc, mae))
        # 模型保存
        if mae < best_mae:
            torch.save(net.state_dict(), Config.get_pth_path())
            best_mae = mae


if __name__ == '__main__':
    train()

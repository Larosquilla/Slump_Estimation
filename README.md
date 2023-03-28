desc: main.py
author: yilin
date:2022-09-18

# title:参数说明

data 数据集目录
-a 网络架构名称
-j 数据载入工作核
--epochs 150
--start-epoch 重启后继续
-b batch_size
--lr 学习率
--momentum

--wd weight-dacay
-p print frequency
--resume checkpoint文件路径
-e 在验证集上验证
--pretrained 使用预训练的模型
--word-size 分布式训练结点
--seed 初始化随机种子


# 模型训练与测试

已有的model是用data parallel分布式训练的,暂时不知道怎么载入

model用的是model.py里面的resnet18

python -a resnet18 --epochs 150 --pretrained '' --gpu 0 

python torch_onnx_verify.py 



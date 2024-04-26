import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from pathlib import Path
from rich import print
import sys
import random

# 定义三个类别
CLSAA = [0, 1, 2]  # 0 松 1 正常 2 紧

# 设置logger日志输出格式， 不包含年月日
# 设置日志格式
logger.remove()  # 清除默认配置
logger.add(sys.stdout, format="{time:HH:mm:ss} {message}")


# 归一化函数
def normalize_mfccs(mfccs):
    mfccs_mean = mfccs.mean(axis=1, keepdims=True)
    mfccs_std = mfccs.std(axis=1, keepdims=True)
    normalized_mfccs = (mfccs - mfccs_mean) / mfccs_std
    return normalized_mfccs


# 音频加载和特征提取函数
# sr: 采样率
# n_mfcc: MFCC的数量
def load_audio_features(file_path: str, sr: int = 22050, n_mfcc: int = 40):
    audio, sr = librosa.load(file_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # print("mfccs.shape", mfccs.shape)  # (40 , 431)
    # 0是MFCC特征的数量，431是时间帧的数量

    # 归一化
    mfccs = normalize_mfccs(mfccs)

    return mfccs


# 读取文件夹下所有音频文件
def read_audio_files(folder_path: str):

    audio_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                audio_paths.append(os.path.join(root, file))
    return audio_paths


# 数据集类
class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels):
        self.audio_paths: list[str] = audio_paths
        self.labels: list[int] = labels

    def __len__(self):
        assert len(self.audio_paths) == len(self.labels), "数据和标签数量不一致"
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        label = self.labels[idx]
        mfcc = load_audio_features(path)
        mfcc = mfcc.T  # 转置 (40, 431) -> (431, 40)
        # print("mfcc.shape", mfcc.shape)
        return torch.Tensor(mfcc), torch.tensor(label, dtype=torch.long)


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        # input_size: 每个时间步输入的特征维度
        # hidden_size: LSTM的隐藏状态维度
        # num_layers: LSTM的层数
        # batch_first=True: 输入数据的形状为(batch_size, seq_len, input_size)
        # 指定输入和输出张量的第一个维度是批量大小
        self.lstm = nn.LSTM(
            input_size=40, hidden_size=64, num_layers=2, batch_first=True
        )
        # 定义一个全连接层，将LSTM的最后一个隐藏状态（64维）映射到3个输出类别
        self.fc = nn.Linear(64, len(CLSAA))

    def forward(self, x):

        # 运行LSTM层，它返回最终的隐藏状态h_n和细胞状态

        _, (h_n, _) = self.lstm(x)

        # 将LSTM的最后一个隐藏状态通过全连接层进行转换
        x = self.fc(h_n[-1])

        # 应用log softmax函数，计算每个类别的概率的对数。dim=1表示沿着类别维度进行softmax

        return F.log_softmax(x, dim=1)


# 训练模型函数
# model: 模型
# dataloader: 数据加载器
# criterion: 损失函数
# optimizer: 优化器
# num_epochs: 训练的轮数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (features, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 梯度清零
            outputs = model(features)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            total_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()

            if (i + 1) % 5 == 0:
                print(
                    f"训练周期 [{epoch+1}/{num_epochs}], 步骤 [{i+1}/{len(dataloader)}], 步骤损失: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"训练周期 [{epoch+1}/{num_epochs}], 平均损失: {avg_loss:.4f}")

    print("训练完成")

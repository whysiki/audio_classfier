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
import noisereduce as nr
from multiprocessing import Pool, cpu_count
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# 定义三个类别
CLSAA = [0, 1, 2]  # 0 松 1 正常 2 紧
CLSAA_DICT = {0: "松", 1: "正常", 2: "紧"}


# 保存日志文件,以追加模式，每天一个文件
logger.add("logs/{time:YYYY-MM-DD-HH}.log", rotation="1 day", encoding="utf-8")


# 归一化函数
def normalize_mfccs(mfccs) -> np.ndarray:
    mfccs_mean = mfccs.mean(axis=1, keepdims=True)
    mfccs_std = mfccs.std(axis=1, keepdims=True)
    normalized_mfccs = (mfccs - mfccs_mean) / mfccs_std
    return normalized_mfccs


# 音频加载和特征提取函数
# sr: 采样率
# n_mfcc: MFCC的数量
def load_audio_features(
    file_path: str, sr: int = 22050, n_mfcc: int = 40, augment: bool = False
) -> np.ndarray:

    audio, sr = librosa.load(file_path, sr=sr)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # logger.info("mfccs.shape", mfccs.shape)  # (40 , 431)

    # 0是MFCC特征的数量，431是时间帧的数量

    # 归一化
    mfccs = normalize_mfccs(mfccs)

    # 转置 (40, 431) -> (431, 40)
    mfccs = mfccs.T

    return mfccs


# 读取文件夹下所有音频文件
def read_audio_files(folder_path: str) -> list[str]:

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

        self.label_list = [
            torch.tensor(label, dtype=torch.long) for label in self.labels
        ]

        with Pool(int(cpu_count())) as p:
            self.feature_list = p.starmap(
                load_audio_features,
                [(path, 22050, 40, True) for path in self.audio_paths],
            )

    def __len__(self) -> int:
        assert len(self.audio_paths) == len(self.labels), "数据和标签数量不一致"
        return len(self.audio_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        return self.feature_list[idx], self.label_list[idx]


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc = nn.Linear(64 * 2, len(CLSAA))  # 输出层，考虑双向LSTM的输出维度

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # 由于使用了双向LSTM，需要将前向和后向的隐藏状态拼接起来
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        x = self.dropout(h_n)  # 应用dropout
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# 训练模型函数
# model: 模型
# dataloader: 数据加载器
# criterion: 损失函数
# optimizer: 优化器
# num_epochs: 训练的轮数
def train_model(
    model,
    dataloader,
    criterion=nn.CrossEntropyLoss(),  # 使用交叉熵损失函数
    optimizer=None,
    num_epochs=30,
    draw_loss=False,
):
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    loss_list = []
    for epoch in range(num_epochs):
        total_loss = 0
        # loss_list = []
        for i, (features, labels) in enumerate(dataloader):
            optimizer.zero_grad()  # 梯度清零
            outputs = model(features)  # 前向传播
            # print("outputs: ", outputs.tolist())
            loss = criterion(outputs, labels)  # 计算损失
            loss_list.append(loss.item())
            total_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()

            if (i + 1) % 5 == 0:
                logger.info(
                    f"epoch [{epoch+1}/{num_epochs}], step [{i+1}/{len(dataloader)}], loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        logger.info(f"epoch [{epoch+1}/{num_epochs}], avg_loss: {avg_loss:.4f}")

    logger.success("训练完成")

    if draw_loss:
        plt.plot(loss_list, label="loss")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        logger.warning("close the plot window to continue...")
        plt.show()


# 一个测试加载器
def test_model(model, dataloader) -> float:
    model.eval()  # 设置模型为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        # 遍历数据加载器 ，获取特征和标签 ，labels是真实标签
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # 获取模型的输出
            outputs = model(features)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            assert predicted.shape == labels.shape, "预测结果和标签数量不一致"

            total += labels.size(0)

            predicted_class = [
                CLSAA_DICT[predicted[i].item()] for i in range(len(predicted))
            ]

            labels_class = [CLSAA_DICT[labels[i].item()] for i in range(len(labels))]

            print(
                "predicted_class: ",
                predicted_class,
            )

            print(
                "labels_class: ",
                labels_class,
            )

            if all(p == l for p, l in zip(predicted_class, labels_class)):
                logger.success("all equal")
            else:
                logger.error("not equal")

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.debug(f"Accuracy on the test set: {accuracy:.2f}%")

    return accuracy


# 给定音频路径获取类型
def get_audio_type(model, audio_path: str) -> str:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = load_audio_features(audio_path)
    outputs = model(features)
    _, predicted = torch.max(outputs.data, 1)
    return CLSAA_DICT[predicted.item()]

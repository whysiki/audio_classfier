import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from pathlib import Path
from rich import print
import sys
import random
from tqdm import tqdm

# import noisereduce as nr
from multiprocessing import Pool, cpu_count, Value

# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from some_tools import (
    # add_tensorboard_image,
    find_similar_segments,
    interpolate_mfcc,
    # normalize_mfccs,
    apply_random_augmentation,
)
from torch.utils.tensorboard import SummaryWriter

# import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义三个类别
CLSAA = [0, 1, 2]  # 0 松 1 正常 2 紧
CLSAA_DICT = {0: "松", 1: "正常", 2: "紧"}
# 插值目标长度
TARGET_LENGTH = 100
# 提取特征数
N_MFCC = 50
# 采样率
SR = None

# 保存日志文件,以追加模式，每天一个文件
logger.add("logs/{time:YYYY-MM-DD-HH}.log", rotation="1 day", encoding="utf-8")


def accept_tuple_argument(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 如果传入的参数只有一个，且是元组
        if len(args) == 1 and isinstance(args[0], tuple):

            result = func(*args[0])  # 解包元组

            if hasattr(func, "tqdm_object") and func.tqdm_object.total is not None:
                func.tqdm_object.update(1)

            return result

        result = func(*args, **kwargs)

        if hasattr(func, "tqdm_object") and func.tqdm_object.total is not None:
            func.tqdm_object.update(1)

        return func(*args, **kwargs)

    return wrapper


@accept_tuple_argument
def load_audio_features(
    file_path: str,
    sr: int,
    n_mfcc: int,
    augment: bool = False,
) -> np.ndarray:
    """
    file_path: 音频文件路径
    sr: 采样率
    n_mfcc: MFCC 特征数量
    augment: 是否数据增强
    """

    audio, sr = librosa.load(file_path, sr=sr)  # Tuple[ndarray, float]

    # 数据增强
    if augment:
        # 随机应用数据增强
        audio = apply_random_augmentation(audio, sr)

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # 去除特征重复的帧

    similar_pairs = find_similar_segments(mfccs)

    original_length = mfccs.shape[1]

    # 去除所有配对的相似帧中索引更大的帧，以保持有序性
    mfccs = np.delete(mfccs, [max(pair) for pair in similar_pairs], axis=1)

    new_length = mfccs.shape[1]

    # logger.info(f"去除重复帧 {original_length - new_length} 个")

    # 归一化
    # mfccs = normalize_mfccs(mfccs)

    return mfccs


# 数据集类
class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, EXTEND_TIMES=3):

        initial_start_time = datetime.datetime.now()
        logger.info("start to initialize dataset")

        assert len(audio_paths) == len(
            labels
        ), "The number of audio files does not match the number of labels."

        self.audio_paths: list[str] = audio_paths
        self.labels: list[int] = labels

        self.label_list = [
            torch.tensor(label, dtype=torch.long) for label in self.labels
        ]

        with ProcessPoolExecutor(max_workers=int(cpu_count())) as executor:
            self.feature_list = []

            # 放大倍数 放大数据集
            EXTEND_TIMES = EXTEND_TIMES

            logger.info(f"amplify dataset {EXTEND_TIMES} magnification times")

            tasks_total = len(audio_paths) * (EXTEND_TIMES + 1)

            for augment in [False] + [True] * EXTEND_TIMES:
                features = list(
                    tqdm(
                        executor.map(
                            load_audio_features,
                            [(path, SR, N_MFCC, augment) for path in audio_paths],
                        ),
                        total=int(tasks_total / (EXTEND_TIMES + 1)),
                        desc="pre-process audio_paths features",
                        smoothing=0.2,
                        unit="file",
                        dynamic_ncols=True,
                        colour="yellow",
                    )
                )
                self.feature_list.extend(features)

            self.label_list.extend(self.label_list * EXTEND_TIMES)

            logger.info(
                f"amplify dataset {EXTEND_TIMES} magnification times, added the features datas be enhanced and shiffted"
            )

        assert len(self.feature_list) == len(
            self.label_list
        ), "The number of features does not match the number of labels."

        target_length = TARGET_LENGTH  # 固定长度

        logger.info(f"target_length: {target_length}")

        # 插值所有 MFCC 矩阵到这个目标长度
        self.feature_list = [
            interpolate_mfcc(mfcc, target_length) for mfcc in self.feature_list
        ]

        logger.info("features interpolation completed")

        # 转置
        self.feature_list = [mfcc.T for mfcc in self.feature_list]

        logger.info("features transpose completed")

        # 转换为张量
        self.feature_list = [
            torch.tensor(mfcc, dtype=torch.float32) for mfcc in self.feature_list
        ]

        logger.info("features to tensor completed")

        logger.info("dataset size : " + str(len(self.label_list)))

        initial_end_time = datetime.datetime.now()

        logger.success(
            f"dataset initialization completed, executed in {initial_end_time - initial_start_time} seconds"
        )

    def __len__(self) -> int:
        len_labels = len(self.label_list)
        len_features = len(self.feature_list)
        assert (
            len_labels == len_features
        ), "The number of labels does not match the number of features"
        return len_labels

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:

        return self.feature_list[idx], self.label_list[idx]


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=N_MFCC,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        # self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.dropout = nn.Dropout(0.6)  # 添加Dropout层
        self.fc = nn.Linear(64 * 2, len(CLSAA))  # 输出层，双向LSTM的输出维度

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # 由于使用了双向LSTM，需要将前向和后向的隐藏状态拼接起来
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        x = self.dropout(h_n)  # 应用dropout
        x = self.fc(x)

        return x


# 记录训练时间
def count_time(tag: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"start to perform :  {tag}")
            start_time = datetime.datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            logger.success(f"{tag} executed in {end_time - start_time} seconds")
            return result

        return wrapper

    return decorator


# 训练模型函数
# model: 模型
# dataloader: 数据加载器
# criterion: 损失函数
# optimizer: 优化器
# num_epochs: 训练的轮数
@count_time("train_model")
def train_model(
    model,
    dataloader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=None,
    num_epochs=40,
    draw_loss=False,
    grad_clip=None,  # 梯度裁剪 超参 float
    patience=6,  # 早停 # 超参 int
    warmup_epochs=5,  # 学习率预热 # 超参 预热的轮数
):
    if not optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 使用 AdamW 优化器
    scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.1
    )  # 学习率调度器
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1
    )  # 学习率预热
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_list = []

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in tqdm(
        range(num_epochs),
        desc="training",
        unit="epoch",
        colour="green",
        smoothing=0.2,
        leave=True,
        dynamic_ncols=True,
    ):

        total_loss = 0
        for i, (features, labels) in tqdm(
            enumerate(dataloader),
            desc="batch",
            unit="batch",
            smoothing=0.2,
            dynamic_ncols=True,
            colour="#00ffff",
            leave=False,
        ):
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()

            if (i + 1) % 5 == 0:
                logger.info(
                    f"epoch [{epoch+1}/{num_epochs}], step [{i+1}/{len(dataloader)}], loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)  # 更新学习率
        if epoch >= warmup_epochs:  # 学习率预热
            warmup_scheduler.step()
        logger.debug(f"epoch [{epoch+1}/{num_epochs}], avg_loss: {avg_loss:.4f}")

        if patience:

            logger.debug(
                f"patience [{epochs_no_improve}/{patience}], best_loss: {best_loss:.4f}"
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    logger.success("Early stopping")
                    break

    logger.success("Training completed")

    if draw_loss:
        writer = SummaryWriter()
        tag = "Loss" + str(datetime.datetime.now())
        for step, loss in enumerate(loss_list):
            writer.add_scalar(tag, loss, step)
        writer.close()


# 一个测试加载器
@count_time("test_model")
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

            assert (
                predicted.shape == labels.shape
            ), "The shape of the predicted and labels is not equal"

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


# 单个音频文件预测
@count_time("get_audio_type")
def get_audio_type(model, audio_path: str, device: torch.device) -> str:
    model.eval()
    with torch.no_grad():
        features = load_audio_features(audio_path)
        # 插值
        features = interpolate_mfcc(features, TARGET_LENGTH)
        features = torch.tensor(features).to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
    return CLSAA_DICT[predicted.item()]

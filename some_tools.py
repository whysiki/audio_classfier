from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import interp1d
import os
import librosa
import random
import librosa.display
import matplotlib.pyplot as plt
from rich import print
from functools import wraps
import datetime
from loguru import logger


# 归一化函数
def normalize_mfccs(mfccs: np.ndarray) -> np.ndarray:
    mfccs_mean = mfccs.mean(axis=1, keepdims=True)
    mfccs_std = mfccs.std(axis=1, keepdims=True)
    normalized_mfccs = (mfccs - mfccs_mean) / mfccs_std
    return normalized_mfccs


def add_tensorboard_image(tensor, name, dataformats="HW"):
    writer = SummaryWriter()
    writer.add_image(name, tensor, dataformats=dataformats)
    writer.close()


# 计算余弦相似度
# mfccs: MFCC特征
# threshold: 阈值
def find_similar_segments(mfccs: np.ndarray, threshold=0.99):
    # 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(mfccs.T)
    similar_pairs = []

    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            if sim_matrix[i, j] > threshold:
                similar_pairs.append((i, j))

    return similar_pairs


# 插值函数, 统一shape
def interpolate_mfcc(mfccs: np.ndarray, target_length: int) -> np.ndarray:
    n_mfcc, original_length = mfccs.shape
    interpolation_function = interp1d(
        np.arange(original_length),
        mfccs,
        # kind="linear",
        # kind="nearest",
        # 立方插值 ，圆滑插值
        kind="cubic",
        axis=1,
        fill_value="extrapolate",
    )
    new_index = np.linspace(0, original_length - 1, target_length)
    new_mfccs = interpolation_function(new_index)
    return new_mfccs


# 读取文件夹下所有音频文件
def read_audio_files(folder_path: str) -> list[str]:

    audio_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                audio_paths.append(os.path.join(root, file))

    assert len(audio_paths) > 0, "没有找到音频文件"

    return audio_paths


# 随机应用数据增强
def apply_random_augmentation(audio: np.ndarray, sr) -> np.ndarray:
    # 数据增强
    # 随机时间拉伸
    def time_stretch(audio) -> np.ndarray:
        stretch_factor = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(y=audio, rate=stretch_factor)

    # 随机音调变换
    def pitch_shift(audio, sr) -> np.ndarray:
        shift_steps = np.random.randint(-5, 5)
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=shift_steps)

    # 随机添加噪声
    def add_noise(audio) -> np.ndarray:
        noise = np.random.normal(0, 0.005, len(audio))
        return audio + noise

    # 随机改变音量
    def change_volume(audio) -> np.ndarray:
        volume = np.random.uniform(0.5, 1.5)
        return audio * volume

    # # 随机移位
    def random_shift(audio) -> np.ndarray:
        shift = np.random.randint(0, len(audio))
        return np.roll(audio, shift)

    # 随机插入静音
    def insert_silence(audio) -> np.ndarray:
        silence_duration = np.random.uniform(0.1, 0.5)
        silence = np.zeros(int(silence_duration * 22050))
        insert_position = np.random.randint(0, len(audio))
        return np.insert(audio, insert_position, silence)

    # 什么都不做
    def nothing(audio) -> np.ndarray:
        return audio

    augmentations = [
        time_stretch,
        pitch_shift,
        add_noise,
        change_volume,
        random_shift,
        nothing,
    ]

    num_augmentations = random.randint(1, len(augmentations))

    for _ in range(num_augmentations):
        augmentation = random.choice(augmentations)
        if augmentation == pitch_shift:
            audio = augmentation(audio, sr)
        else:
            audio = augmentation(audio)

    return audio


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

    # 查看数据类型
    # print(mfccs.dtype) float32

    return mfccs

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import interp1d
import os
import librosa
import random

# 创建一个随机的3x3张量
# tensor = torch.randn(3, 3)

# 创建一个TensorBoard写入器
# writer = SummaryWriter()

# 将张量数据写入TensorBoard
# writer.add_image("Tensor Data", tensor, dataformats="HW")

# 关闭写入器


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
        kind="nearest",
        # 立方插值 ，圆滑插值
        # kind="cubic",
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


# 随机移位
def random_shift(audio) -> np.ndarray:
    shift = np.random.randint(0, len(audio))
    return np.roll(audio, shift)


# 随机应用数据增强
def apply_random_augmentation(audio: np.ndarray, sr) -> np.ndarray:

    augmentations = [time_stretch, pitch_shift, add_noise, change_volume, random_shift]

    num_augmentations = random.randint(1, len(augmentations))

    for _ in range(num_augmentations):
        augmentation = random.choice(augmentations)
        if augmentation == pitch_shift:
            audio = augmentation(audio, sr)
        else:
            audio = augmentation(audio)

    return audio


# tensorboard --logdir=runs

#


# if __name__ == "__main__":
#     test_tensor = torch.randn(3, 3, 3)
#     add_tensorboard_image(test_tensor, "test_tensor2", "CHW")

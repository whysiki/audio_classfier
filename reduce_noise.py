import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pre_process import load_audio_features

# from some_tools import *
# from uuid import uuid4
# import datetime
import matplotlib.pyplot as plt


def plot_mfccs(mfccs_frame_1, mfccs_frame_2):  # , origin_frame_1, origin_frame_2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Plot first frame
    axs[0].imshow(mfccs_frame_1, cmap="hot", interpolation="nearest")
    axs[0].set_title("MFCCs Frame 1")

    # Plot second frame
    axs[1].imshow(mfccs_frame_2, cmap="hot", interpolation="nearest")
    axs[1].set_title("MFCCs Frame 2")

    # 设置刻度1-40
    for ax in axs:
        ax.set_yticks(np.arange(0, 40, 1))
        # ax.set_xticks(np.arange(0, 1, 1))

    # 把图放在屏幕中间

    plt.tight_layout()
    plt.show(block=False)
    # plt.pause(2)
    plt.close()


def find_similar_segments(mfccs, threshold=0.99):
    # 计算余弦相似度矩阵
    sim_matrix = cosine_similarity(mfccs.T)
    similar_pairs = []

    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            if sim_matrix[i, j] > threshold:
                similar_pairs.append((i, j))

    return similar_pairs


if __name__ == "__main__":
    # 示例
    test_audio_path = r"声纹采集数据\敲弹条\WJ-7\松\第四组\WJ-7-连续五个-9.wav"
    mfccs = load_audio_features(test_audio_path)
    similar_pairs = find_similar_segments(mfccs)

    # 分别输出每对相似帧的 MFCC
    for pair in similar_pairs:
        frame_1, frame_2 = pair
        mfccs_frame_1 = mfccs[:, frame_1 : frame_1 + 1]
        mfccs_frame_2 = mfccs[:, frame_2 : frame_2 + 1]
        print(f"Pair ({frame_1}, {frame_2}):")
        # tag = str(uuid4())

        # plot_mfccs(mfccs_frame_1, mfccs_frame_2)

        # 去除相似帧中一个
        # mfccs = np.delete(mfccs, frame_2, axis=1)

        # print(f"MFCCs Frame {frame_1}: {mfccs_frame_1.tolist()}")
        # print(f"MFCCs Frame {frame_2}: {mfccs_frame_1.tolist()}")

    # 去除所有配对的相似帧中一个
    mfccs = np.delete(mfccs, [pair[0] for pair in similar_pairs], axis=1)

# 可视化
#
# # Plot MFCCs
# plt.figure(figsize=(18, 5))  # (width, height)
# plt.imshow(mfccs, cmap="gray", interpolation="nearest")
# plt.title("MFCCs")
# plt.yticks(np.arange(0, 40, 5))
# plt.xticks(np.arange(0, mfccs.shape[1], 20))
# plt.tight_layout()
# plt.show()


# print(f"MFCCs: {mfccs.tolist()}")

# shape
# print(f"MFCCs shape: {mfccs.shape}")


# sm = find_similar_segments(mfccs)

# print(f"Similar segments: {sm}")


# 'hot': 从黑色过渡到红色，然后到橙色，最后到黄色
# 'cool': 从青色过渡到洋红色
# 'gray': 从黑色过渡到白色的灰度映射
# 'viridis': 一种在色彩感知上均匀的映射，从深紫色过渡到黄色
# 'plasma': 一种在色彩感知上均匀的映射，从深紫色过渡到橙色
# 'inferno': 一种在色彩感知上均匀的映射，从黑色过渡到深红色
# 'magma': 一种在色彩感知上均匀的映射，从黑色过渡到淡粉色

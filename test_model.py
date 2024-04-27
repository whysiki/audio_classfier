# # # LSTM模型
# # class AudioClassifier(nn.Module):
# #     def __init__(self):
# #         super(AudioClassifier, self).__init__()
# #         # input_size: 每个时间步输入的特征维度
# #         # hidden_size: LSTM的隐藏状态维度
# #         # num_layers: LSTM的层数
# #         # batch_first=True: 输入数据的形状为(batch_size, seq_len, input_size)
# #         # 指定输入和输出张量的第一个维度是批量大小
# #         self.lstm = nn.LSTM(
# #             input_size=40, hidden_size=64, num_layers=2, batch_first=True
# #         )
# #         # 定义一个全连接层，将LSTM的最后一个隐藏状态（64维）映射到3个输出类别
# #         self.fc = nn.Linear(64, len(CLSAA))

# #     def forward(self, x) -> torch.Tensor:

# #         # 运行LSTM层，它返回最终的隐藏状态h_n和细胞状态

# #         _, (h_n, _) = self.lstm(x)

# #         # 将LSTM的最后一个隐藏状态通过全连接层进行转换
# #         x = self.fc(h_n[-1])

# #         # 使用log_softmax函数将输出转换为概率

# #         x = F.log_softmax(x, dim=1)

# #         return x


# # class AudioClassifier(nn.Module):
# #     def __init__(self):
# #         super(AudioClassifier, self).__init__()
# #         self.conv1 = nn.Conv1d(40, 64, kernel_size=5, stride=2)
# #         self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2)
# #         self.lstm = nn.LSTM(
# #             input_size=128, hidden_size=128, num_layers=3, batch_first=True
# #         )
# #         self.dropout = nn.Dropout(0.5)
# #         self.fc = nn.Linear(128, len(CLSAA))

# #     def forward(self, x) -> torch.Tensor:
# #         x = x.transpose(1, 2)  # Switch time and feature dimensions for Conv1d
# #         x = F.relu(self.conv1(x))
# #         x = F.relu(self.conv2(x))
# #         x = x.transpose(1, 2)  # Switch time and channel dimensions for LSTM
# #         _, (h_n, _) = self.lstm(x)
# #         x = self.dropout(h_n[-1])
# #         x = self.fc(x)
# #         x = F.log_softmax(x, dim=1)
# #         return x


# # class AudioClassifier(nn.Module):
# #     def __init__(self):
# #         super(AudioClassifier, self).__init__()
# #         self.lstm = nn.LSTM(
# #             input_size=40,
# #             hidden_size=64,
# #             num_layers=2,
# #             batch_first=True,
# #             bidirectional=True,
# #         )
# #         self.dropout = nn.Dropout(0.5)  # 添加Dropout层
# #         self.fc = nn.Linear(64 * 2, len(CLSAA))  # 输出层，考虑双向LSTM的输出维度

# #     def forward(self, x):
# #         _, (h_n, _) = self.lstm(x)
# #         # 由于使用了双向LSTM，需要将前向和后向的隐藏状态拼接起来
# #         h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
# #         x = self.dropout(h_n)  # 应用dropout
# #         x = self.fc(x)
# #         return F.log_softmax(x, dim=1)

# # class AudioClassifier(nn.Module):
# #     def __init__(self):
# #         super(AudioClassifier, self).__init__()
# #         self.lstm = nn.LSTM(
# #             input_size=40,
# #             hidden_size=64,  # 每个方向有128个隐藏单位
# #             num_layers=2,
# #             batch_first=True,
# #             bidirectional=True,
# #         )
# #         self.dropout = nn.Dropout(0.5)
# #         self.bn = nn.BatchNorm1d(64 * 2)  # 双向LSTM，所以是256
# #         self.fc1 = nn.Linear(64 * 2, 128)  # 256是双向LSTM的输出
# #         self.fc2 = nn.Linear(128, len(CLSAA))

# #     def forward(self, x):
# #         _, (h_n, _) = self.lstm(x)
# #         h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
# #         h_n = self.bn(h_n)
# #         x = F.relu(self.fc1(h_n))
# #         x = self.dropout(x)
# #         x = self.fc2(x)
# #         return F.log_softmax(x, dim=1)
# # 降噪 , 假设整个音频文件包含噪声
# # audio = nr.reduce_noise(y=audio, sr=sr)
# # if augment:
# #     augment = Compose(
# #         [
# #             AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
# #             TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
# #             PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
# #             Shift(
# #                 min_shift=-0.5,
# #                 max_shift=0.5,
# #                 shift_unit="fraction",
# #                 rollover=True,
# #                 p=0.5,
# #             ),
# #         ]
# #     )
# #     audio = augment(samples=audio, sample_rate=sr)

# # import torch
# # import torch.nn as nn
# # from torchviz import make_dot


# # class AudioClassifier(nn.Module):
# #     def __init__(self):
# #         super(AudioClassifier, self).__init__()
# #         self.lstm = nn.LSTM(
# #             input_size=431,
# #             hidden_size=64,
# #             num_layers=2,
# #             batch_first=True,
# #             bidirectional=True,
# #         )
# #         self.dropout = nn.Dropout(0.5)  # 添加Dropout层
# #         self.fc = nn.Linear(64 * 2, len(CLSAA))  # 输出层，双向LSTM的输出维度

# #     def forward(self, x):
# #         _, (h_n, _) = self.lstm(x)
# #         # 由于使用了双向LSTM，需要将前向和后向的隐藏状态拼接起来
# #         h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
# #         x = self.dropout(h_n)  # 应用dropout
# #         x = self.fc(x)
# #         return F.log_softmax(x, dim=1)


# # model = AudioClassifier()


# # # 这里假设你的输入张量的形状为(batch_size, sequence_length, input_dim)
# # input_tensor = torch.randn(1, 10, 431)  # 假设 batch_size = 1, sequence_length = 10
# # # 前向传播是神经网络中的一个过程，用于将输入数据通过网络层进行处理，最终得到模型的输出结果。
# # # 在前向传播过程中，数据从输入层经过各个隐藏层，最终到达输出层。
# # # 每个隐藏层都会对输入数据进行加权求和、非线性变换等操作，以提取特征和逐步转化数据
# # output_tensor = model(input_tensor)


# # # 将输入张量传递给make_dot函数，它会自动构建模型图
# # make_dot(output_tensor, params=dict(model.named_parameters()))


# def load_audio_features(
#     file_path: str,
#     sr: int = 22050,
#     n_mfcc: int = 40,
#     augment: bool = False,
#     # target_length: int = 431,
# ) -> np.ndarray:

#     audio, sr = librosa.load(file_path, sr=sr)

#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

#     # 去除重复的帧

#     similar_pairs = find_similar_segments(mfccs)

#     # 去除所有配对的相似帧中一个
#     mfccs = np.delete(mfccs, [pair[0] for pair in similar_pairs], axis=1)

#     # logger.info("mfccs.shape", mfccs.shape)  # (40 , 431)

#     # audio = nr.reduce_noise(y=audio, sr=sr)

#     # 40是MFCC特征的数量，431是时间帧的数量

#     # 归一化
#     # mfccs = normalize_mfccs(mfccs)

#     # print("mfccs: ", mfccs.tolist())

#     # 转置 (40, 431) -> (431, 40)
#     # mfccs = mfccs.T

#     # add_tensorboard_image(mfccs, "mfccsT")

#     return mfccs

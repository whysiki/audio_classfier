from pre_process import *


# CLSAA = [0, 1, 2]  # 0 松 1 正常 2 紧

# 声纹采集数据\敲螺栓
normal_audio_paths = []
lose_audio_paths = []
tight_audio_paths = []

for path in [
    r"声纹采集数据\敲螺栓\WJ-7\有噪声\正常",
]:
    normal_audio_paths += read_audio_files(path)

for path in [
    r"声纹采集数据\敲螺栓\WJ-7\有噪声\松",
]:
    lose_audio_paths += read_audio_files(path)

for path in [
    r"声纹采集数据\敲螺栓\WJ-7\有噪声\紧",
]:
    tight_audio_paths += read_audio_files(path)


audio_paths: list[str] = normal_audio_paths + lose_audio_paths + tight_audio_paths

# test
# print(audio_paths)

audio_paths_labels = (
    [1] * len(normal_audio_paths)
    + [0] * len(lose_audio_paths)
    + [2] * len(tight_audio_paths)
)

zip_list = list(zip(audio_paths, audio_paths_labels))
zip_list_audio_paths = [x[0] for x in zip_list]
zip_list_labels = [x[1] for x in zip_list]


dataset = AudioDataset(zip_list_audio_paths, zip_list_labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# 实例化模型

model = AudioClassifier()

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用Adam优化器
# lr: 学习率 设为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
train_model(model, dataloader, criterion, optimizer, draw_loss=True)

# 保存模型

torch.save(model.state_dict(), Path("model") / "hit_luo_s_WJ-7_model.pth")


# 测试模型

# 加载模型
model = AudioClassifier()
model.load_state_dict(torch.load(Path("model") / "hit_luo_s_WJ-7_model.pth"))


# 实例化数据集和数据加载器
# 随机抽取10个样本
shamped_list = random.sample(zip_list, 5)
shamped_list_audio_paths = [x[0] for x in shamped_list]
shamped_list_labels = [x[1] for x in shamped_list]

# 实例化数据集和数据加载器
test_dataset = AudioDataset(zip_list_audio_paths, zip_list_labels)

test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# 测试模型

test_model(model=model, dataloader=test_dataloader)

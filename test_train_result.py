from pre_process import *

normal_audio_paths = []
lose_audio_paths = []
tight_audio_paths = []
# 零件种类
Part_Type = "敲弹条"
# 零件编号
Part_No = "WJ-8"

for path in [r"声纹采集数据\{}\{}\正常".format(Part_Type, Part_No)]:
    normal_audio_paths += read_audio_files(path)

for path in [r"声纹采集数据\{}\{}\松".format(Part_Type, Part_No)]:
    lose_audio_paths += read_audio_files(path)

for path in [r"声纹采集数据\{}\{}\紧".format(Part_Type, Part_No)]:
    tight_audio_paths += read_audio_files(path)


# 验证任务
# 已经训练的样本识别准确率需要为100%
# 数据集
# 训练数据集和测试数据集相等
def test_train_result():
    # CLSAA = [0, 1, 2]  # 0 松 1 正常 2 紧

    # 声纹采集数据\敲螺栓

    batch_size = 4

    audio_paths: list[str] = normal_audio_paths + lose_audio_paths + tight_audio_paths

    audio_paths_labels = (
        [1] * len(normal_audio_paths)
        + [0] * len(lose_audio_paths)
        + [2] * len(tight_audio_paths)
    )

    zip_list = list(zip(audio_paths, audio_paths_labels))  # [:2]
    zip_list_audio_paths = [x[0] for x in zip_list]
    zip_list_labels = [x[1] for x in zip_list]

    dataset = AudioDataset(zip_list_audio_paths, zip_list_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 实例化模型

    model = AudioClassifier()

    # 训练模型
    train_model(model, dataloader, draw_loss=True)

    # 保存模型

    # torch.save(model.state_dict(), Path("model") / "hit_luo_s_WJ-7_model.pth")

    # 测试模型

    # 加载模型
    # model = AudioClassifier()
    # model.load_state_dict(torch.load(Path("model") / "hit_luo_s_WJ-7_model.pth"))

    # 实例化数据集和数据加载器
    # 随机抽取10个样本
    # shamped_list = random.sample(zip_list, 5)
    # shamped_list_audio_paths = [x[0] for x in shamped_list]
    # shamped_list_labels = [x[1] for x in shamped_list]

    # 实例化数据集和数据加载器
    test_dataset = AudioDataset(zip_list_audio_paths, zip_list_labels)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 测试模型

    accuracy = test_model(model=model, dataloader=test_dataloader)

    assert accuracy == 100.0, f"accuracy: {accuracy}"


def get_audio_paths_labels(
    normal_audio_paths, lose_audio_paths, tight_audio_paths
) -> tuple[list[str], list[int]]:

    audio_paths: list[str] = normal_audio_paths + lose_audio_paths + tight_audio_paths

    audio_paths_labels = (
        [1] * len(normal_audio_paths)
        + [0] * len(lose_audio_paths)
        + [2] * len(tight_audio_paths)
    )

    zip_list = list(zip(audio_paths, audio_paths_labels))

    zip_list_audio_paths = [x[0] for x in zip_list]
    zip_list_labels = [x[1] for x in zip_list]

    return zip_list_audio_paths, zip_list_labels


# 预测任务 交叉验证
# 分割数据集
# 测试集和训练集不相等，且不参与训练 打乱数据集
def test_train_result_slpit_origian():

    global normal_audio_paths, lose_audio_paths, tight_audio_paths

    normal_audio_paths = random.sample(normal_audio_paths, len(normal_audio_paths))
    lose_audio_paths = random.sample(lose_audio_paths, len(lose_audio_paths))
    tight_audio_paths = random.sample(tight_audio_paths, len(tight_audio_paths))

    n_border = int(len(normal_audio_paths) * 4 / 5)
    l_border = int(len(lose_audio_paths) * 4 / 5)
    t_border = int(len(tight_audio_paths) * 4 / 5)

    left_zip_list_audio_paths, left_zip_list_labels = get_audio_paths_labels(
        normal_audio_paths=normal_audio_paths[:n_border],
        lose_audio_paths=lose_audio_paths[:l_border],
        tight_audio_paths=tight_audio_paths[:t_border],
    )

    right_zip_list_audio_paths, right_zip_list_labels = get_audio_paths_labels(
        normal_audio_paths=normal_audio_paths[n_border:],
        lose_audio_paths=lose_audio_paths[l_border:],
        tight_audio_paths=tight_audio_paths[t_border:],
    )

    dataset = AudioDataset(left_zip_list_audio_paths, left_zip_list_labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 实例化模型

    model = AudioClassifier()

    # 训练模型
    train_model(model, dataloader, draw_loss=True)

    # 保存模型

    # torch.save(model.state_dict(), Path("model") / "hit_luo_s_WJ-7_model.pth")

    # 测试模型

    # 加载模型
    # model = AudioClassifier()
    # model.load_state_dict(torch.load(Path("model") / "hit_luo_s_WJ-7_model.pth"))

    # 实例化数据集和数据加载器
    test_dataset = AudioDataset(right_zip_list_audio_paths, right_zip_list_labels)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 测试模型

    accuracy = test_model(model=model, dataloader=test_dataloader)

    return accuracy

    # assert accuracy == 100.0, f"accuracy: {accuracy}"


if __name__ == "__main__":
    # test_train_result()
    test_accuracy_list = []
    for i in range(10):
        pass
        accuracy = test_train_result_slpit_origian()
        test_accuracy_list.append(accuracy)

    plt.figure()
    plt.plot(test_accuracy_list)
    plt.savefig("test_accuracy_list.png")
    plt.show()

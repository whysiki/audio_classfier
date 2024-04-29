from pre_process import *
from some_tools import *


## 测试数据

normal_audio_paths = []
lose_audio_paths = []
tight_audio_paths = []
# 零件种类
Part_Type = "敲弹条"
# 零件编号
Part_No = "WJ-7"

for path in [r"声纹采集数据\{}\{}\正常".format(Part_Type, Part_No)]:
    normal_audio_paths += read_audio_files(path)

for path in [r"声纹采集数据\{}\{}\松".format(Part_Type, Part_No)]:
    lose_audio_paths += read_audio_files(path)

for path in [r"声纹采集数据\{}\{}\紧".format(Part_Type, Part_No)]:
    tight_audio_paths += read_audio_files(path)


#
#
# 验证任务
# 已经训练的样本识别准确率需要为100%
# 数据集
# 训练数据集和测试数据集相等
@count_time(
    tag=f"样本数量: {len(normal_audio_paths) + len(lose_audio_paths) + len(tight_audio_paths)}, 验证任务: 已经训练的样本识别准确率需要为100%"
)
def test_train_result(
    normal_audio_paths: str = normal_audio_paths,
    lose_audio_paths: str = lose_audio_paths,
    tight_audio_paths: str = tight_audio_paths,
) -> AudioClassifier:

    # normal_audio_paths = []
    # lose_audio_paths = []
    # tight_audio_paths = []
    # # 零件种类
    # Part_Type = Part_Type
    # # 零件编号
    # Part_No = Part_No

    # for path in [r"声纹采集数据\{}\{}\正常".format(Part_Type, Part_No)]:
    #     normal_audio_paths += read_audio_files(path)

    # for path in [r"声纹采集数据\{}\{}\松".format(Part_Type, Part_No)]:
    #     lose_audio_paths += read_audio_files(path)

    # for path in [r"声纹采集数据\{}\{}\紧".format(Part_Type, Part_No)]:
    #     tight_audio_paths += read_audio_files(path)

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

    dataset = AudioDataset(zip_list_audio_paths, zip_list_labels, EXTEND_TIMES=3)
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
    test_dataset = AudioDataset(zip_list_audio_paths, zip_list_labels, EXTEND_TIMES=0)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 测试模型

    accuracy = test_model(model=model, dataloader=test_dataloader)

    assert accuracy == 100.0, f"accuracy: {accuracy}"

    return model


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

    # logger.info(f"样本数量: {len(zip_list)}")

    zip_list_audio_paths = [x[0] for x in zip_list]
    zip_list_labels = [x[1] for x in zip_list]

    assert len(zip_list_audio_paths) == len(zip_list_labels), "样本-标签 数量不一致"

    return zip_list_audio_paths, zip_list_labels


# 预测任务 交叉验证
# 分割数据集
# 测试集和训练集不相等，且不参与训练 打乱数据集
# 由于样本数量较少，所以采用了数据集放大的方式， 原始数据集+ 数据增强数据集
def test_train_result_slpit_origian():

    global normal_audio_paths, lose_audio_paths, tight_audio_paths

    normal_audio_paths = random.sample(normal_audio_paths, len(normal_audio_paths))
    lose_audio_paths = random.sample(lose_audio_paths, len(lose_audio_paths))
    tight_audio_paths = random.sample(tight_audio_paths, len(tight_audio_paths))

    n_border = int(len(normal_audio_paths) * 7 / 10)
    l_border = int(len(lose_audio_paths) * 7 / 10)
    t_border = int(len(tight_audio_paths) * 7 / 10)

    logger.info(f"训练样本数量: {n_border + l_border + t_border}")
    logger.info(
        f"测试样本数量: {len(normal_audio_paths) + len(lose_audio_paths) + len(tight_audio_paths) - n_border - l_border - t_border}"
    )

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

    dataset = AudioDataset(
        left_zip_list_audio_paths, left_zip_list_labels, EXTEND_TIMES=3
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

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
    test_dataset = AudioDataset(
        right_zip_list_audio_paths, right_zip_list_labels, EXTEND_TIMES=0
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 测试模型

    accuracy = test_model(model=model, dataloader=test_dataloader)

    return accuracy

    # assert accuracy == 100.0, f"accuracy: {accuracy}"


if __name__ == "__main__":
    # pass
    # model = test_train_result()
    test_accuracy_list = []
    writer = SummaryWriter()
    tag = "test_predict_accuracy__" + str(datetime.datetime.now())
    for i in range(20):
        pass
        logger.info(f"交叉验证--第{i+1}次测试")
        accuracy = test_train_result_slpit_origian()
        test_accuracy_list.append(accuracy)
        writer.add_scalar(tag, accuracy, i)
        logger.success(f"交叉验证--第{i+1}次测试完成")
    writer.close()

    # logger.info(f"test_accuracy_list: {test_accuracy_list}")
    # logger.info(f"平均准确率: {sum(test_accuracy_list) / len(test_accuracy_list)}")

    # plt.figure()
    # plt.plot(test_accuracy_list)
    # plt.savefig("test_accuracy_list.png")
    # plt.xlabel("time")
    # plt.ylabel("accuracy")

    # 添加到tensorboard
    # writer = SummaryWriter()
    # for i in range(10):
    #     writer.add_scalar("accuracy", test_accuracy_list[i], i)
    # plt.show()

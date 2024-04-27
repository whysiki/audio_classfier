from torch.utils.tensorboard import SummaryWriter
import torch

# 创建一个随机的3x3张量
# tensor = torch.randn(3, 3)

# 创建一个TensorBoard写入器
# writer = SummaryWriter()

# 将张量数据写入TensorBoard
# writer.add_image("Tensor Data", tensor, dataformats="HW")

# 关闭写入器


def add_tensorboard_image(tensor, name, dataformats="HW"):
    writer = SummaryWriter()
    writer.add_image(name, tensor, dataformats=dataformats)
    writer.close()


# tensorboard --logdir=runs

#


if __name__ == "__main__":
    test_tensor = torch.randn(3, 3, 3)
    add_tensorboard_image(test_tensor, "test_tensor2", "CHW")

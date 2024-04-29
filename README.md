### 一个对敲击零件发出的声音来判断零件是否合格的机器学习项目

- MFCC 特征提取
- 基于双向 LTSM 网络
- 音频三分类问题

CLSAA = [0, 1, 2] # 0 松 1 正常 2 紧
CLSAA_DICT = {0: "松", 1: "正常", 2: "紧"}

音频预处理和模型定义相关代码在 `pre_process.py`

### 训练和预处理

### 模型性能

- 对于参与训练的零件，识别准确率达到 100%
- 对于未参与训练的零件，预测准确率较稳定到 100%，但是有时会出现 90% 左右的准确率

- 最近 10 次训练测试的预测准确率 `test_train_result.py`
  - 100 个样本的交叉验证，比例 7:3

```python
if __name__ == "__main__":
    # test_train_result()
    test_accuracy_list = []
    for i in range(10):
        pass
        accuracy = test_train_result_slpit_origian()
        test_accuracy_list.append(accuracy)

    plt.figure()
    plt.plot(test_accuracy_list)
    plt.show()
```

![alt text](images/README/image-1.png)

- Loss

![alt text](images/README/image-2.png)

<!-- - 模型结构 -->

<!-- ![alt text](audio_classifier.png) -->

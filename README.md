# README

基于MFCC特征构建单核GMM的0-9独立词语音识别，MFCC，GMM，sklearn，Isolated word recognition。

## 实验内容

基于0-9数字语音数据集，使用GMM对10个数字逐一建模，对输入的音频进行分类，识别语音中表达的数字。

## 实验方案

1. MFCC 特征提取-librosa
2. 端点检测
3. GMM模型构建（本算法构建的GMM使用diagonal 方差）
4. sklearn GMM调包测试

## Preinstallation

使用pip可安装使用的所有依赖

## 使用手册

运行endpoint_audio.py可对records文件夹下所有文件进行端点检测，并生成相应处理后的文件存储在./Processed_records文件夹下。

模型训练和预测默认以当前最优模型组合进行，即以Librosa提取13维mfcc特征构建GMM。

```python
python endpoint_audio.py	#生成端点检测后的音频，默认使用records文件夹下的数据
# 使用自己编写的GMM模型训练和测试数据
python Recognizer.py -t train -c GMM -i "records/*/" -m model.out	
python Recognizer.py -t predict -c GMM -i "records/*/" -m model.out
# 使用Sklearn中的GMM模型训练和测试数据
python Recognizer.py -t train -c Sklearn -i "records/*/" -m model.out
python Recognizer.py -t predict -c GMM -i "records/*/" -m model.out
```


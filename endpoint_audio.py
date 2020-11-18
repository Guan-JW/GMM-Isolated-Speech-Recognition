# -*- coding: utf-8 -*-
import wave
import os
import numpy as np
from endpoint_detect import EndPointDetect

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--maxnum", help="max digit", type=int, default=9)
parser.add_argument("-c", "--count", help="examples for each digit", type=int, default=20)
parser.add_argument("-d", "--datapath", help="dir of data", type=str, default="./records/")

parser.add_argument("-f", "--framerate", help="max digit", type=int, default=16000)
parser.add_argument("-ch", "--channels", help="采样频率 8000 or 16000", type=int, default=1)
parser.add_argument("-s", "--sampwidth", help="max digit", type=int, default=2)
args = parser.parse_args()

maxnum = args.maxnum  # max digit
count = args.count  # examples for each digit
# 存储成 wav 文件的参数
framerate = args.framerate  # 采样频率 8000 or 16000
channels = args.channels  # 声道数
sampwidth = args.sampwidth  # 采样字节 1 or 2
datapath = args.datapath

def Mkdir(path):
    '''create directory'''
    if not os.path.exists(path):
        os.mkdir(path)


# 将语音文件存储成 wav 格式
def save_wave_file(filename, data):
    '''save the data to the wavfile'''
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)   # 声道
    wf.setsampwidth(sampwidth)  # 采样字节 1 or 2
    wf.setframerate(framerate)  # 采样频率 8000 or 16000
    wf.writeframes(b"".join(data))
    wf.close()


Mkdir('Processed_records')

for i in tqdm(range(maxnum + 1)):
    Mkdir('Processed_records/digit_'+str(i))
    for j in range(count):
        f = wave.open(datapath + "digit_" + str(i) + "/" + str(j + 1) + '_' + str(i) + ".wav", "rb")
        params = f.getparams()  # 一次性返回所有的WAV文件的格式信息
        nchannels, sampwidth, framerate, nframes = params[:4]   # nframes 采样点数目
        str_data = f.readframes(nframes)  # str_data 是二进制字符串， readframes() 按照采样点读取数据
        # print(str_data)
        # 以上可以直接写成 str_data = f.readframes(f.getnframes())

        # 转成二字节数组形式（每个采样点占两个字节）
        wave_data = np.fromstring(str_data, dtype=np.short)
        # print(wave_data)
        # print(str(i + 1) + "-" + str(j + 1) + " 采样点数目：" + str(len(wave_data)))          #输出应为采样点数目
        # print(str(i) + "-" + str(j + 1) + " 采样点数目：" + str(len(wave_data)))          #输出应为采样点数目

        f.close()

        # 端点检测
        end_point_detect = EndPointDetect(wave_data)
        N = end_point_detect.wave_data_detected
        # 输出为 wav 格式
        m = 0
        while m < len(N):
            save_wave_file("./Processed_records/digit_" + str(i) + "/" + str(j + 1) + '_' + str(i) + ".wav",
                           wave_data[N[m] * 256: N[m + 1] * 256])   # 256个采样点为一帧
            m = m + 2

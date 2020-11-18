
from collections import defaultdict
import os
import math
import mfcc_features as mf
import itertools
import glob
from GMMs import GMM_Model
from gmm import GMM_Model_sklearn
import sys
import numpy as np
import argparse


def train_model(input_dirs, output_name, model, train_test_split=0.25, method='Librosa'):
    '''
    Train GMM models for each number

    :param input_dirs:  directory for input audios
    :param output_name: path for the output model name
    :param train_test_split: the ratio of test audios
    :param method: method for extracting mfcc features. - Librosa / Speech_feature
    :return: None
    '''

    features = defaultdict(list)  # save features for each number 0-9

    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs)) # 铺平成一个list
    dirs = [d for d in dirs if os.path.isdir(d)]

    if len(dirs) == 0:
        print ("No valid directory found!")
        sys.exit(1)

    train_size = math.floor(20 * (1 - train_test_split))
    print("Extracting MFCC features...")
    for d in dirs:
        label = (d.split('\\')[1]).split('_')[1]
        for i in range(1, train_size + 1):
            wav_path = d + str(i) + '_' + label + '.wav'
            # # print(fs, signal)
            if method == 'Speech_feature':
                mfcc_feature = mf.get_mfcc_feature(wav_path)
            else:
                mfcc_feature = mf.lib_mfcc_feature(wav_path)
            features[int(label)].extend(mfcc_feature)
    print("Successfully extracted MFCC features\n")
    print("Constructing GMM Models...")
    if model=='Sklearn':
        gmm_model = GMM_Model_sklearn(features)
    else:
        gmm_model = GMM_Model(features)
    gmm_model.train()
    gmm_model.dump(output_name)


def predict(input_dirs, model_path, train_test_split=0.25, method='Librosa'):
    '''
    Use GMM models to predict the input audios.

    :param input_dirs: directory for input audios.
    :param model_path: path for the pre-trained GMM model.
    :param train_test_split: ratio of test audios.
    :param method: method for extracting mfcc features. -Librosa / Python Speech Feature (shoud be same as the pre-trained model)
    :return: None
    '''

    gmm_model = GMM_Model.load(model_path)

    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs)) # 铺平成一个list
    dirs = [d for d in dirs if os.path.isdir(d)]

    if len(dirs) == 0:
        print ("No valid directory found!")
        sys.exit(1)

    train_size = math.floor(20 * (1 - train_test_split))
    true_labels = []
    pred_labels = []
    for d in dirs:
        label = (d.split('\\')[1]).split('_')[1]
        for i in range(train_size + 1, 21):
            wav_path = d + str(i) + '_' + label + '.wav'
            # fs, signal = read_audio(wav_path)
            true_labels.append(int(label))
            # label_pred = gmm_model.predict(fs, signal)
            label_pred = gmm_model.predict(wav_path, method=method)
            pred_labels.append(label_pred)
            print("label, pred = ({}, {})".format(label, label_pred))

    precision = np.sum(np.array(true_labels) == np.array(pred_labels)) / len(true_labels)
    print("precision: %f" %(precision))



def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
            Wav files in each input directory will be labeled as the basename of the directory.
            Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.
            Examples:
                Train (enroll a list of isolated audio files, e.g.'records/*/', with wav files under corresponding directories):
                ./speaker-recognition.py -t train -i "/tmp/person* ./mary" -m model.out
                Predict (predict the speaker of all wav files):
                ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
            """
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "train" or "predict"',
                       required=True)

    parser.add_argument('-c', '--method',
    					help='Model to use. Either "GMM" or "Sklearn"',
    					required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to train)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in train) or use(in predict)',
                       required=True)

    ret = parser.parse_args()
    return ret


if __name__ == "__main__":
    # Python speech feature mfcc 13维 34%
    # train_model('records/*/', 'model1.out', method='Speech_feature')
    # predict('records/*/', 'model1.out', method='Speech_feature')

    # Python speech feature mfcc 39维 32%
    # train_model('records/*/', 'model1.out', method='Speech_feature')
    # predict('records/*/', 'model1.out', method='Speech_feature')

    # Python speech feature 10条测试  29%
    # train_model('records/*/', 'model2.out',train_test_split=0.5, method='Speech_feature')
    # predict('records/*/', 'model2.out', train_test_split=0.5, method='Speech_feature')

    # librosa mfcc 39维 42%
    # train_model('records/*/', 'model3.out')
    # predict('records/*/', 'model3.out')

    # librosa 6条测试 43.3333%
    # train_model('records/*/', 'model4.out', train_test_split=0.3)
    # predict('records/*/', 'model4.out', train_test_split=0.3)

    # librosa 10条测试 42%
    # train_model('records/*/', 'model5.out', train_test_split=0.5)
    # predict('records/*/', 'model5.out', train_test_split=0.5)

    # 含端点检测 python speech features mfcc 13维 5条测试 48%
    # train_model('Processed_records/*/', 'model6.out', method='Speech_feature')
    # predict('Processed_records/*/', 'model6.out', method='Speech_feature')

    # 含端点检测 python speech features mfcc 39维 5条测试 58%
    # train_model('Processed_records/*/', 'model7.out', method='Speech_feature')
    # predict('Processed_records/*/', 'model7.out', method='Speech_feature')

    # 含端点检测 python speech features mfcc 39维 10条测试 48%
    # train_model('Processed_records/*/', 'model7.out', method='Speech_feature')
    # predict('Processed_records/*/', 'model7.out', method='Speech_feature')

    # 含端点检测 python speech features mfcc 13维 6条测试 48.333%
    # train_model('Processed_records/*/', 'model8.out', method='Speech_feature', train_test_split=0.3)
    # predict('Processed_records/*/', 'model8.out', method='Speech_feature', train_test_split=0.3)

    # 含端点检测 librosa 使用mode='nearest',否则音频时长过短，滤波较大，无法正常识别，但损失精度，24%
    # train_model('Processed_records/*/', 'model9.out')
    # predict('Processed_records/*/', 'model9.out')

    # librosa mfcc 13维 5条测试 76%
    # train_model('records/*/', 'model10.out')
    # predict('records/*/', 'model10.out')

    # Librosa mfcc 13维 10条测试 65%
    # train_model('records/*/', 'model11.out',  train_test_split=0.5)
    # predict('records/*/', 'model11.out',  train_test_split=0.5)

    # endpoint Librosa mfcc 13维 5条测试 68%
    # train_model('Processed_records/*/', 'model12.out')
    # predict('Processed_records/*/', 'model12.out')

    # endpoint Librosa mfcc 13维 10条测试 56%
    # train_model('Processed_records/*/', 'model13.out', train_test_split=0.5)
    # predict('Processed_records/*/', 'model13.out', train_test_split=0.5)

    # librosa mfcc 20维 5条测试 62%
    # train_model('records/*/', 'model14.out')
    # predict('records/*/', 'model14.out')

    args = get_args()

    task = args.task
    if task == 'train':
        train_model(args.input, args.model, args.method ,method='Librosa')
    elif task == 'predict':
        predict(args.input, args.model, method='Librosa')

    # python Recognizer.py -t train -i "records/*/" -m model.out
    # python Recognizer.py -t predict -i "records/*/" -m model.out
    #
    # python Recognizer.py -t train -i "Processed_records/*/" -m model.out
    # python Recognizer.py - t predict - i "Processed_records/*/" - m model.out
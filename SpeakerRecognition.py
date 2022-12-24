# 테스트 파일 경로 여기서 수정하시면 됩니다.
test_path="C:/Users/user0425/Desktop/DigitalSound/1.wav"
#"C:/Users/user0425/Desktop/DigitalSound/2.wav"
#"C:/Users/user/PycharmProjects/pythonProject1/test/F3tr (mp3cut.net).wav"

import numpy as np
import pandas as pd
import librosa #0.8.1 교수님 쓰시는 버전
import os
import sys
from sklearn.mixture import GaussianMixture

def load(audio_path):
  audio1, sr1 = librosa.load(audio_path , sr=16000)
  mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr1, n_mels=24, n_fft=512, hop_length=512)
  mfcc2 = pd.DataFrame(mfcc1)
  mfcc2 = mfcc2.T
  return mfcc2

test=load(test_path)
train_f1 = load("C:/Users/user0425/Desktop/DigitalSound/F1tr.wav")
train_f2 = load("C:/Users/user0425/Desktop/DigitalSound/F2tr.wav")
train_f3 = load("C:/Users/user0425/Desktop/DigitalSound/F3tr.wav")
train_m1 = load("C:/Users/user0425/Desktop/DigitalSound/M1tr.wav")
train_m2 = load("C:/Users/user0425/Desktop/DigitalSound/M2tr.wav")
train_m3 = load("C:/Users/user0425/Desktop/DigitalSound/M3tr.wav")

gmm_f1 = GaussianMixture(n_components=4, random_state=0).fit(train_f1)
gmm_f2 = GaussianMixture(n_components=4, random_state=0).fit(train_f2)
gmm_f3 = GaussianMixture(n_components=4, random_state=0).fit(train_f3)
gmm_m1 = GaussianMixture(n_components=4, random_state=0).fit(train_m1)
gmm_m2 = GaussianMixture(n_components=4, random_state=0).fit(train_m2)
gmm_m3 = GaussianMixture(n_components=4, random_state=0).fit(train_m3)

scores=[gmm_f1.score(test),gmm_f2.score(test),gmm_f3.score(test),gmm_m1.score(test),gmm_m2.score(test),gmm_m3.score(test)]
labels=['F1','F2','F3','M1','M2','M3']

result = scores.index(max(scores))

f = open("result.txt","a", encoding='utf8')
f.write(labels[result]+'\n')
f.close()
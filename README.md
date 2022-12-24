# Speaker-Recognition

## About
> Speaker recognition among three female and male voices

<br>

## Procedure
> Data Load → MFCC extraction → create each GMM model → test data load → gmm.score → decision

<br>

## Data Information
> TrainingData : 10sec of audio data <br> Training : 8sec of audio data(used to create models) <br> Test : 2sec of audio data(used to test the model)

<br>

## GMM
> 가우시안 혼합 모델 (GMM)은 주어진 표본 데이터 집합의 분포밀도를 단 하나의 확률밀도함수로 모델링하는 방법을 개선한 밀도 추정 방법으로 복수 개의 가우시안 확률밀도함수로 데이터의 분포를 모델링 하는 방법이다.
# Voice-Converter
> Module for freely modifying or controlling voice





## Introduction

> 본 directory 학습내용의 출처는 [FutureSkill](https://futureskill.io/)의 [Contents](https://futureskill.io/content/aae67c6d-236e-4667-8b5e-e59668f0b562)입니다.



[목소리를 변형하는 모듈]을 통해 목소리를 자유 자재로 변형하거나 제어해보자.



#### 학습 목표

- **[최종] 입력 음성에 원하는 변형을 가한 출력 음성 생성 모듈**
- *(목표 1) 음성의 요소 이해 및 추출 (크기, 빠르기)*
- *(목표 2) pyworld vocoder를 이용한 음정 변환*
- *(목표 3) pysox 모듈을 이용한 빠르기, 감정 변환*
- *(목표 4) noise, reverb 적용 연습*



음성의 여러가지 특징적인 feature들을 이해하고, 그것을 제어할 수 있도록 하는 방법을 이해함으로서

향후 음성 데이터 전처리, augmentation 등 과정을 보다 편하게 진행할 수 있도록 돕는 과정이다.



## Contents



1. 음성 데이터 살펴보기
2. 음성 데이터 분석하기 - rms, pitch
3. 음성 데이터 분석하기 - timing
4. 음성 데이터 재합성하기 - 1
5. 음성 데이터 재합성하기 - 2
6. pitch 제어하기 - 높낮이
7. pitch 제어하기 - 감정
8. 빠르기 제어하기
9. Noise 제어하기
10. Reverb 제어하기
11. 목소리 변환기 만들기



---

#### Reference

- [Future Skill](https://futureskill.io/)

#### Editor

- [**Colab**](https://colab.research.google.com/) / PyCharm

# 목소리 변환기
> 음성을 자유롭게 수정하고 제어하는 모듈 [English/영어](./README.md)



## 개요

> 본 레포 학습내용의 출처는 [FutureSkill](https://futureskill.io/)의 [Contents](https://futureskill.io/content/aae67c6d-236e-4667-8b5e-e59668f0b562)입니다.

[목소리를 변형하는 모듈]을 통해 목소리를 자유 자재로 변형하거나 제어해보자.



#### 목표

- **[최종] 입력 음성에 원하는 변형을 가한 출력 음성 생성 모듈**
- *(목표 1) 음성의 요소 이해 및 추출 (크기, 빠르기)*
- *(목표 2) pyworld vocoder를 이용한 음정 변환*
- *(목표 3) pysox 모듈을 이용한 빠르기, 감정 변환*
- *(목표 4) noise, reverb 적용 연습*

음성의 여러가지 특징적인 feature들을 이해하고, 그것을 제어할 수 있도록 하는 방법을 이해함으로서 향후 음성 데이터 전처리, augmentation 등 과정을 보다 편하게 진행할 수 있도록 돕는 과정이다. 전체적인 개발은 Google Colab에서 진행했다.



## Contents

1. [음성 데이터 살펴보기](#1-음성-데이터-살펴보기)
2. [음성 데이터 분석하기 (rms, pitch)](#2-음성-데이터-분석하기-rms-pitch)
3. 음성 데이터 분석하기 (timing)
4. 음성 데이터 재합성하기 1
5. 음성 데이터 재합성하기 2
6. pitch 제어하기 (높낮이)
7. pitch 제어하기 (감정)
8. 빠르기 제어하기
9. Noise 제어하기
10. Reverb 제어하기
11. 목소리 변환기 만들기





> Last Update: 2021.08.10.

### 1. 음성 데이터 살펴보기

> *"그는 괜찮은 척 하려고 애쓰는 것 같았다."*  [speech.wav](./speech.wav) 파일을 재생하면 나오는 음성이다.

음성은 사람이 발성기관을 통해 내는 소리를 의미한다. 넓은 의미의 음성에는 말소리, 노래와 같은 언어적 음성 뿐만 아니라 웃음소리, 기침 소리 같은 비언어적 음성이 포함된다. 언어적 음성, 그 중에서도 일반적인 대화에서 사용되는 음성을 중점적으로 살펴보자. 이번 콘텐츠에서 지속적으로 사용할 라이브러리를 정리하면 다음과 같다.

~~~python
import numpy as np  
import librosa
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import librosa.display
~~~

나의 경우, Colab에서 음원을 들어보기 위해 아래와 같은 방법을 사용한다. 이후에도 간편하게 활용할 일이 매우 많았다.

~~~python
import IPython.display as ipd
ipd.Audio('speech.wav')
~~~

괄호 내부에 듣고자 하는 음원의 경로를 입력하면 output이 예쁘게 출력된다.



먼저 Spectrogram() 함수를 선언한다.

~~~python
def Spectrogram(wav):
		stft = librosa.stft(wav)
		stft = np.abs(stft)
		return stft
~~~

이후 [**librosa**](https://librosa.org/doc/latest/index.html) library를 활용해 `speech.wav` 음원을 다운로드해 불러오고,  

[**matplotlib**](https://matplotlib.org/) library를 이용해 waveform과 spectrogram을 그려본다.

~~~python
audio, sr = librosa.load('speech.wav', sr=None)
spectrogram = np.log(Spectrogram(audio)+1e-5)

plt.figure(figsize=(16,9))
plt.subplot(2,1,1)
plt.plot(audio)
plt.xlim([0,len(audio)])
plt.title('waveform')
plt.subplot(2,1,2)
librosa.display.specshow(spectrogram)
plt.title('spectrogram')
plt.tight_layout()
plt.savefig('example1_output.png')
# plt.close()
~~~

![1_1](./outputs/example1_output.png)





### 2. 음성 데이터 분석하기 (rms, pitch)

waveform은 시간에 따른 압력 정보를 표현한 그래프이므로, 어떤 시점에서 마이크로 얼만큼의 압력이 인가되었는지를 알려주고 있는 정보이다.

spectrogram은 이러한 waveform에 Short-time Fourier transform(STFT)을 적용한 결과로, 작은 window로 제한된 시간에 따라 해당 순간에 각 주파수 성분이 얼만큼의 강도로 표현되어있는지를 시각화해서 확인할 수 있는 정보이다. 두 정보 모두 아주 자세한 정보를 포함하고 있는 셈이지만, 우리는 보다 직관적인 정보를 필요로 하는 경우가 있다.

이번엔 우리에게 조금 더 친숙한 개념인, **세기(rms)**와 **음정(pitch)** 정보를 음원으로부터 출력하는 과정을 진행한다.

**세기(rms)**는 waveform 내부에서 특정 window 안에 있는 모든 압력 값들의 제곱 평균에 root 를 취한 값이다. 일반적으로 우리가 '큰 소리'를 녹음한다면 이 값은 커지는 경향이 있고, 반대로 작은 소리는 작은 경향이 있다. `librosa.feature.rms` 함수를 통해 계산이 가능하다.

**음정(pitch)**은 발화에 포함되어있는 시간에 따른 음의 높낮이를 의미한다. 높은 음일수록 높은 주파수라고 생각하면 된다. `pyworld` 라이브러리를 통해 pitch sequence를 출력할 수 있다.

먼저 [**pyworld**](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) library가 필요하다.

~~~python
!pip install pyworld
~~~

pyworld를 설치한 후 코드는 다음과 같다.

 ~~~python
 import pyworld as pw 
 
 audio, sr = librosa.load('speech.wav', sr=None)
 spectrogram = np.log(np.abs(librosa.core.stft(audio))+1e-5)
 
 audio = np.asarray(audio, dtype='float64')
 
 _f0, t = pw.dio(audio, sr) # raw pitch extractor
 f0 = pw.stonemask(audio, _f0, t, sr) # pitch refinement
 rms = librosa.feature.rms(audio) 
 plt.figure(figsize=(16,9))
 
 plt.subplot(4,1,1)
 plt.plot(audio)
 plt.xlim([0,len(audio)])
 plt.title('waveform')
 
 plt.subplot(4,1,2)
 librosa.display.specshow(spectrogram)
 plt.title('spectrogram')
 
 plt.subplot(4,1,3)
 plt.plot(rms[0])
 plt.xlim([0,len(rms[0])])
 plt.title('rms')
 
 plt.subplot(4,1,4)
 plt.plot(f0)
 plt.xlim([0,len(f0)])
 plt.title('pitch')
 
 plt.tight_layout()
 plt.savefig('example2_output.png')
 plt.close()
 ~~~

![2_1](./outputs/example2_output.png)





### 3. 음성 데이터 분석하기 - timing

앞선 예제를 통해서 우리는 주어진 음성에서 각 시간에 따른 파형, spectrogram, rms, pitch 정보를 출력하는 방법을 익혔다. 따라서 음원과 특정 시점이 주어지면, 해당 시점에서 여러가지 정보들을 파악할 수 있게 되었다.

그렇다면 '시점' 자체에 대한 정보는 어떻게 파악할 수 있을까? 예컨데 주어진 음원은 "**그는 괜찮은척 하려고 애쓰는것 같았다."** 라는 문자열을 발화하고 있는데, 이 중 `그` 나 `괜` 이 몇 초부터 시작되는지는 어떻게 파악할 수 있을까?

사실 음성으로부터 그에 포함된 언어적 표현에 대한 정보를 추측해내는 것은 `Speech-to-text`, `Automatic Speech Recognition`, `Speech-Text Alignment` 등 다양한 이름과 분야로 연구되고있는 쉽지 않은 주제이다. 그러나 우리는 가장 간단한 접근법 중 하나로, 주어진 시퀀스에서 어떤 새로운 event가 등장한 `onset` 시점을 판단해보는 실험을 해볼 수 있다.

`onset`은 '**the beginning of a musical note or other sound**' 으로 정의될 수 있으며, librosa 라이브러리에서는 waveform 으로부터 onset이 등장했을법한 시점을 계산해주는 [`onset_detect`](https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html#librosa.onset.onset_detect) 함수를 제공하고 있다.

  

일단 onset을 찾은 후 matplotlib을 활용해 plot에 선을 그어 구분했다.

~~~python
import soundfile as sf # 오디오파일 생성을 위한 library

audio, sr = librosa.load('speech.wav', sr=None)
onsets = librosa.onset.onset_detect(audio, sr=44100, hop_length=512)
onsets = librosa.frames_to_time(onsets, sr=44100, hop_length=512)
# print(onset_times) # 확인용 

plt.figure(figsize=(16,9))
plt.subplot(2,1,1)
plt.plot(audio)
plt.xlim([0,len(audio)])

for item in onsets:
    plt.axvline(x=(int)(item*sr), c='r')
    # print(item) # 확인용
    # 세로 선을 x 위치에 그려주는 함수. 
plt.title('waveform')

plt.subplot(2,1,2)
librosa.display.specshow(spectrogram)
for item in onsets:
    plt.axvline(x=(int)(item*sr)/512, c='r')
plt.title('spectrogram')

plt.tight_layout()
plt.savefig('example3_output.png')
plt.close()

for i in range(len(onsets[:-1])):
    sf.write('example3_output_'+str(i).zfill(2)+'.wav', audio[(int)(onsets[i]*sr):(int)(onsets[i+1]*sr)], sr)
~~~

![3_1](./outputs/example3_output.png)

위 코드로 생성된 16개의 음원을 들어보면 `그`, `는`, `괜`, `찮`, `은`, ... 정확하지는 않지만 어느 정도 음절마다 끊겨서 저장된 것을 확인할 수 있다.





### 4. 음성 데이터 재합성하기 1

앞선 일련의 예제를 통해서 우리는 주어진 음성으로 부터 여러 feature를 '추출' 하는 방법을 익혔다. 하지만 우리의 최종 목표는 변형된 음성을 '생성'해내는 모듈을 디자인하는 것이다.

일반적으로 목소리를 변형시키는 과정은 아래와 같이 생각해볼 수 있다.

`음성` → `(1)feature 추출` → `(2)feature 변형` → `(3)feature 재합성` → `변형된 음성`

현재까지 우리는 (1)에 해당하는 과정을 진행했으며, 성공적인 모듈 디자인을 위해서는 (2)와 (3)에 대해서 파악해야 한다.

`(1)feature 추출`과 `(3)feature 재합성`이 가능하기 위해서는 추출 연산의 역과정이 정의될 수 있어야 한다. 대표적으로 Short time fourier transform은 Inverse short time fourier transform을 정의할 수 있는 변형이기 때문에 reconstruction이 가능하다.

~~~python
audio, sr = librosa.load('speech.wav', sr=None)
spectrogram = librosa.core.stft(audio)
 
audio_recon = librosa.istft(spectrogram, hop_length=512)

sf.write('speech_recon.wav', audio_recon, sr)
~~~

위와 같이 주어진 음원에서 Spectrogram을 추출하고, 해당 Spectrogram을 통해 다시 복원된 wave를 생성하여 저장할 수 있다.





### 5. 음성 데이터 재합성하기 2

Short time fourier transform으로 변형된 feature에는 우리가 원하는 feature 변형의 과정을 진행하기가 쉽지 않다. 예컨데 음정을 올리고 싶은 경우, 우리는 STFT feature에 어떠한 변환과정을 진행해야 하는지 판단하기 어렵다. 이런 한계점을 극복하기 위해서 우리는 pyworld vocoder를 활용할 수 있다.

pyworld vocoder는 [world vocoder](https://pdfs.semanticscholar.org/560a/be3b4482335a93df309cb6a0185ccc3ebd8e.pdf?_ga=2.93225115.742467816.1601196298-72658375.1564975111)의 python 버전 구현체이며, 논문에 따르면 주어진 waveform을 세개의 feature (`Fundamental frequency`, `Spectral Envelope`, `Aperiodic parameter`)로 분리하는 세개의 알고리즘과, 이 세개의 feature를 다시 waveform으로 합성하는 synthesis 알고리즘으로 구성되어있다.

따라서 우리는 이 **추출-재합성 알고리즘**을 사용하면, pitch 정보를 원하는대로 변형한 재합성이 가능하다.

pyworld 라이브러리를 활용하여 주어진 음원에서 `f0`, `spectral envelope`, `aperiodic parameter`를 추출하여 그림을 그리고, 이를 재합성해서 reconstructed wave를 생성해보자.

~~~python
import pyworld as pw

audio, sr = librosa.load('speech.wav', sr=None, dtype='float64')

# raw pitch extractor
_f0, t = pw.dio(audio, sr)
# pitch refinement
f0 = pw.stonemask(audio, _f0, t, sr)
# extract smoothed spectrogram
sp = pw.cheaptrick(audio, _f0, t, sr)
# extract aperiodicity
ap = pw.d4c(audio, f0, t, sr)

y = pw.synthesize(f0, sp, ap, sr)
sf.write('speech_recon_pyworld.wav', y, sr)

plt.figure(figsize=(16,9))
plt.subplot(4,1,1)
plt.plot(audio)
plt.xlim([0,len(audio)])
plt.title('waveform')
plt.subplot(4,1,2)
librosa.display.specshow(np.log(sp.T+1e-5))
plt.title('sp')
plt.subplot(4,1,3)
librosa.display.specshow(np.log(ap.T+1e-5))
plt.title('ap')
plt.subplot(4,1,4)
plt.plot(f0)
plt.xlim([0,len(f0)])
plt.title('pitch')

plt.tight_layout()
plt.savefig('example4_output.png')
plt.close()
~~~

추출-재합성 과정에서 힘든 부분이 있을 경우 [pyworld document](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)를 참고하면 큰 도움이 된다.



### 6. pitch 제어하기 (높낮이)





### 7. pitch 제어하기 (감정)



### 8. 빠르기 제어하기



### 9. Noise 제어하기



### 10. Reverb 제어하기



### 11. 목소리 변환기 만들기













---

#### Reference

- [Future Skill](https://futureskill.io/)

#### Editor

- [**Colab**](https://colab.research.google.com/) / PyCharm

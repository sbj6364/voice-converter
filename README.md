# Voice Converter
> Module for freely modifying or controlling voice  
> [한국어/Korean](./README_KR.md)



## Introduction

> The source of this repository is [Contents](https://futureskill.io/content/aae67c6d-236e-4667-8b5e-e59668f0b562) in [FutureSkill](https://futureskill.io/).

Transforming/controlling voice freely through **Voice-Transforming-Module**.



#### Goals

- <u>[Final] Creating a module that produces a voice with the desired variation on the input voice</u>
- (Goal 1) Understanding and extracting elements of speech (size, speed)
- (Goal 2) **Tone conversion** using *pyworld vocoder*
- (Goal 3) **Speed(tempo), emotional conversion** with *pysox module*
- (Goal 4) Applying **noise**, **reverb**

By understanding various characteristic features of the voice and how to control them, it is a step that makes the process of preprocessing and augmentation easier in the future. The overall development was carried out at *Google Colab.*



## Contents

1. [Examining Voice Data](#1-examining-voice-data)
2. [Analyzing Voice Data (rms, pitch)](#2-analyzing-voice-data-rms-pitch)
3. [Analyzing Voice Data (timing)]( #3-analyzing-voice-data-timing)
4. [Reconstruction 1](#4-reconstruction-1)
5. [Reconstruction 2](#5-reconstruction-2)
6. [Pitch Control (key change)]
7. [Pitch Control (emotional)]
8. [Speed Control]
9. [Noise Control]
10. [Reverb Control]
11. [Voice Converter]





> Last Update: 2021.08.12.

### 1. Examining Voice Data

> *"그는 괜찮은 척 하려고 애쓰는 것 같았다."*  It's a Korean sentence which is in [speech.wav](./speech.wav) file.

**Voice** means the sound a person makes through the vocal organs. Voices include *verbal voices* such as speech and singing, as well as *nonverbal voices* such as laughter and coughing. Let's focus on verbal speech, especially the voice used in general conversations. The libraries that will be used continuously in this content are as follows.

~~~python
import numpy as np  
import librosa
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import librosa.display
~~~

And In my case, I use the following method to listen to audio in Colab. There will be many situations where it will be used conveniently afterwards.

~~~python
import IPython.display as ipd
ipd.Audio('speech.wav')
~~~

If you enter the sound path you want to hear in the bracket, player will be output beautifully.



First, define the Spectrogram() function.

~~~python
def Spectrogram(wav):
		stft = librosa.stft(wav)
		stft = np.abs(stft)
		return stft
~~~



Then load the file [`speech.wav`](./speech.wav) using the [**librosa**](https://librosa.org/doc/latest/index.html) library  
and draw the waveform, spectrogram of it using the [**matplotlib**](https://matplotlib.org/) library

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





### 2. Analyzing Voice Data (rms, pitch)

**Waveform** is a graph of pressure information over time, indicating how much pressure was applied to the microphone at some point.

As a result of the Short-Time Fourier Transform (STFT) application to these waveforms, it is possible to visualize how strong each frequency component is represented at that moment, depending on the time limit of the small window. Both information contains very detailed information, but we sometimes need more intuitive information.



This time, we will proceed with the process of outputting **rms** and **pitch** information, which are more familiar concepts to us.

**RMS (Strength)** is the value taken root from the square mean of all pressure values within a particular window inside the waveform. In general, this value tends to increase if we record a 'big sound', whereas a small sound tends to be small. It can be calculated through the `librosa.feature.rms` function.

**Pitch** means the height of a sound over time that is included in the speech. The higher the pitch, the higher the frequency. Pitch sequence can be printed through the `pyworld` library.



So we need [**pyworld**](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder) library.

~~~python
!pip install pyworld
~~~

After installing pyworld, code is as follows.

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





### 3. Analyzing Voice Data (timing)

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





### 4. Reconstruction 1

need to be updated



### 5. Reconstruction 2





### 6. Pitch Control (key change)





### 7. Pitch Control (emotional)





### 8. Speed Control





### 9. Noise Control





### 10. Reverb Control





### 11. Voice Converter







---

#### Reference

- [Future Skill](https://futureskill.io/)

#### Editor

- [**Colab**](https://colab.research.google.com/) / PyCharm

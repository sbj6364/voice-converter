# Voice Converter
> Module for freely modifying or controlling voice  
> [한국어/Korean](./README_KR.md)



## Introduction

> The source of this repository is [Contents](https://futureskill.io/content/aae67c6d-236e-4667-8b5e-e59668f0b562) in [FutureSkill](https://futureskill.io/).

Transforming/controlling voice freely through **Voice-Transforming-Module**. This file can be viewed as a review note that carefully summarizes the content, process and study results.



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
4. [Reconstruction 1](#4-reconstruction1)
5. [Reconstruction 2](#5-reconstruction2)
6. [Pitch Control (key change)](#6-pitch-control-key-change)
7. [Pitch Control (emotional)](#7-pitch-control-emotional)
8. [Speed Control](#8-speed-control)
9. [Noise Control](#9-noise-control)
10. [Reverb Control](#10-reverb-control)
11. [Voice Converter](#11-voice-converter)





> Last Update: 2021.08.15.

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

Previously we learned how to output waveform, spectrum, rms, and pitch information over each time period from a given voice. Thus, given the sound source and specific time points, various information can be identified at that time.

So how do we get the information about the **timing**? For example, the given sound source emits a string in Korean that says, **"그는 괜찮은척 하려고 애쓰는것 같았다"**, which is translated into English as **"He seemed to be trying to pretend to be okay."** How do we figure out when the syllables such as `He` or `seemed`  begin?

In fact, it is not easy to guess the information about the linguistic expression from speech, which is studied by various names and fields such as `Speech-to-text`, `Automatic Speech Recognition`, and `Speech-Text Alignment`. However, as one of the simplest approaches, we can experiment with determining when a new event appears in a given sequence.

`onset` can be defined as '**the begining of a musical note or other sound**', and the librosa library provides an [`onset_detect`](https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html#librosa.onset.onset_detect) function that calculates when an onset might have appeared from the waveform.

Once find the onset, we can use the matplotlib to draw lines on the plot.

~~~python
import soundfile as sf # Library for creating audio files

audio, sr = librosa.load('speech.wav', sr=None)
onsets = librosa.onset.onset_detect(audio, sr=44100, hop_length=512)
onsets = librosa.frames_to_time(onsets, sr=44100, hop_length=512)
# print(onset_times) # check

plt.figure(figsize=(16,9))
plt.subplot(2,1,1)
plt.plot(audio)
plt.xlim([0,len(audio)])

for item in onsets:
    plt.axvline(x=(int)(item*sr), c='r')
    # print(item) # check
    # A function that draws a vertical line in the x-position
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

By listening to the 16 sound sources generated by the above code, you can see that they are stored, although not very accurate, but broken every syllable.



### 4. Reconstruction1

Through a series of previous examples, we learned how to 'extract' multiple features from a given voice. However, our final goal is to design a module that 'creates' a modified voice.

In general, the process of converting the voice can be considered as follows.

`voice` → `(1)feature extract` → `(2)feature convert` → `(3)feature reconstruct` → `converted voice`

We have conducted a process corresponding to (1) and need to understand (2) and (3) for successful module design.

For `(1)feature extract ` and  `(3)feature reconstruct` to be possible, the inverse process of the extraction operation must be defined. Typically Short Time Fourier Transform(STFT) is a variant that can define Inverse Short Time Fourier Transform(ISTFT), so reconstruction is possible.

~~~python
audio, sr = librosa.load('speech.wav', sr=None)
spectrogram = librosa.core.stft(audio)
 
audio_recon = librosa.istft(spectrogram, hop_length=512)

sf.write('speech_recon.wav', audio_recon, sr)
~~~

As shown above, a Spectrogram can be extracted from a given sound source, and a restored wave can be created and stored again through the Spectrogram.



### 5. Reconstruction2

It is not easy to proceed with the feature deformation process that we want with STFT. For example, if we want to raise the pitch, it is difficult for us to determine what kind of conversion process we should proceed with to STFT feature. To overcome these limitations, we use pyworld vocoder.

pyworld vocoder is a python version implementation of the  [world vocoder](https://pdfs.semanticscholar.org/560a/be3b4482335a93df309cb6a0185ccc3ebd8e.pdf?_ga=2.93225115.742467816.1601196298-72658375.1564975111). According to the paper, it consists of three algorithms that separate a given waveform into three features (`Fundamental frequency`, `Spectral Envelope`, `Aperiodic parameter`) and a synthis algorithm that reconstruct these three features back into waveform.

Thus, we can use this **extraction-reconstruction** algorithm to reconstruct pitch information as desired.

Using the pyworld library, we can extract `f0`, `spectral envelope`, and `periodic parameter` from a given sound source and recreate them to create a reconstructed wave.

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

![4_1](./outputs/example4_output.png)

If there are some difficulties, refer to [pyworld document](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder).



### 6. Pitch Control (key change)

**Pitch** can be expressed in *Hz* units of how many times it vibrates per second, and can be expressed in units of one key, one octave, and so on in music. An octave high note has a frequency value of twice that of its original frequency, and an octave consists of 12 scales. Using this, the following code can be written:

~~~python
def key_change(audio, key):
    _f0, t = pw.dio(audio, sr)   
    f0 = pw.stonemask(audio, _f0, t, sr) 
 
    f0 = f0 + (f0 * key / 12)
    # An octave high note has a frequency value of twice that of its original frequency
		# and an octave consists of 12 scales.
    # Therefore, add 'key' scales of f0 divided by 12 to the existing frequency f0.
 
    sp = pw.cheaptrick(audio, f0, t, sr) 
    ap = pw.d4c(audio, f0, t, sr)        
    y = pw.synthesize(f0, sp, ap, sr)
    return y

sf.write('speech_up1.wav', key_change(audio, 1), sr)
sf.write('speech_up2.wav', key_change(audio, 2), sr)
sf.write('speech_up3.wav', key_change(audio, 3), sr)
 
sf.write('speech_down1.wav', key_change(audio, -1), sr)
sf.write('speech_down2.wav', key_change(audio, -2), sr)
sf.write('speech_down3.wav', key_change(audio, -3), sr)
~~~

By listening to the generated file, you can see that the voice is adjusted as if the key was raised and lowered in a karaoke room.

As such, we can control pitch information from a given music source through pyworld vocoder. It is possible to extract `f0` from the music `y`, generate `f0` that is modified as desired, and then reconstruct `y` through `f0`, `sp`, and `ap`.



### 7. Pitch Control (emotional)

Now we know how to raise or lower the pitch overall in a given sound source. Furthermore, it would be possible to raise and lower it only in certain areas by utilizing it. Beyond these extensions, this time we are going to try to transform the distribution of pitch as a whole. The pitches we obtain through pyworld are sequential data whose values change over time, and are composed of multiple sets of data.

Then there will be an average pitch in this data, and there will also be variances in how far the entire data is from this mean. In the previous problem, it can be understood that the variance was fixed and the pitch of the voice was adjusted by changing the mean value. So, what will happen when you change the variance?

~~~python
def std_change(audio, alpha):
    _f0, t = pw.dio(audio, sr)
    f0 = pw.stonemask(audio, _f0, t, sr)
    mean = np.mean(f0[np.nonzero(f0)])
    std = np.std(f0[np.nonzero(f0)])
    f0[np.nonzero(f0)] -= mean

    f0[np.nonzero(f0)] /= 1
    f0[np.nonzero(f0)] *= alpha

    f0[np.nonzero(f0)] += mean
    sp = pw.cheaptrick(audio, f0, t, sr)
    ap = pw.d4c(audio, f0, t, sr)
    y = pw.synthesize(f0, sp, ap, sr)
    return y

sf.write('speech_up_std.wav', std_change(audio, 2.0), sr)
sf.write('speech_down_std.wav', std_change(audio, 0.5), sr)
~~~

Simply put, if the variance is greater, the change between the data will be greater, so it can be made into a more vibrant or excited voice, and if the variance is smaller, a dull, unchanging, monotonous voice.



### 8. Speed Control

Through previous examples, we looked at generating new voices that transformed the phonetic information extracted from a given audio. This time, we are going to design a module that uses `sox`, a new library, to sequentially process various conversions such as speed, intensity, and etc. as well as pitch.

You may have heard cassette tapes 'fast-forwarded' in the past. This is usually the case even if the video is played at full speed. There are times when a man's voice sounds as thin as the speed of speed increases, making it sound like a woman's voice. This is a change in pitch as speed changes. 

There are two main ways to change the speed of a given sound source. One is to change the speed with the pitch fixed, and the other is to change the pitch and speed together.

The [transformer documentation](https://pysox.readthedocs.io/en/latest/api.html) of pysox module will work this time. First we need to install `sox`.

~~~python
! apt-get install libsox-fmt-all
! apt-get install sox
! pip install sox
~~~

We then create functions for each of the two transformations. The part that is annotated in the middle (the part that defines and returns the `out`) can be unannotated again later when creating the entire voice converter.

~~~python
import sox
def speed_sox(audio, rate):
  tfm = sox.Transformer() # transforming module
  tfm.speed(rate)
  # out = tfm.build_array(input_array=audio, sample_rate_in=sr)
  # return out
  tfm.build_file(output_filepath= 'speech_speed_' + (str)(rate) + '.wav', input_array=audio, sample_rate_in=sr)

def tempo_sox(audio, rate):
  tfm = sox.Transformer()
  tfm.tempo(rate)
  # out = tfm.build_array(input_array=audio, sample_rate_in=sr)
  # return out
  tfm.build_file(output_filepath= 'speech_tempo_' + (str)(rate) + '.wav', input_array=audio, sample_rate_in=sr)

speed_sox(audio, 0.8)
speed_sox(audio, 1.2)

tempo_sox(audio, 0.8)
tempo_sox(audio, 1.2)
~~~

As above, two transformations that change the speed of a given sound source can be found and a function that applies each of them can be written.



### 9. Noise Control

So far, we have learned how to convert the pitch, speed of a given sound source. This time, we want to go beyond the signal itself and add Gaussian noise to the signal. This can vary by task to be applied in the future, but it can also be used as an augmentation method to make the model more robust. Noise is created using `np.random.normal`.

~~~python
def add_noise(audio, rate):
    noise = np.random.normal(0, 1, 155520) # size should be the same as existing sound source's
    return audio + rate*noise

sf.write('speech_noise_0.01.wav', add_noise(audio, 0.01), sr)
sf.write('speech_noise_0.1.wav', add_noise(audio, 0.1), sr)
~~~

The `mean` and `std` of noise are set to 0 and 1 respectively, and input parameters that can control the degree of noise can be used to generate sound sources with different levels of SNR.



### 10. Reverb Control

By #8, we had figured out how to convert the signal itself. #9, added noise to the signal. Similar to the noise added, we would like to practice using impulse response to walk the reverb. Also, reverse can be used as an augmentation method.

The process of reversing in nature can be thought of as a process in which signals reflected in space, in addition to the original signals, are recorded and added through a microphone with a time difference. And to model this experimentally, you can create (or acquire) an impulse response to a particular space and then convert that response to the original signal. The impulse response is the sound received through the microphone when an impulse is generated in that space.

Use [reverb.wav](./reverb.wav) as a given impulse.

~~~python
import scipy

audio, sr = librosa.load('/content/speech.wav')
reverb, sr = librosa.load('/content/reverb.wav')
def apply_reverb(audio, reverb):
    out = scipy.signal.convolve(audio, reverb)
    return out

sf.write('speech_reverb.wav', apply_reverb(audio, reverb), sr)
~~~

The key is to apply the **convolution** operation to speech signals. Refer to [scipy.signal.convolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html). By listening to the output, you can definitely see that the sound contains the reverb.



### 11. Voice Converter

It is a process of coding what you have learned so far as a function. The input conditions are summarized as follows:

- `pitch` : How many keys the pitch should be raised/decreased
- `pitch_var` : How many times the pitch variation should be increased/decreased
- `tempo` : How fast/slow to play
- `change` : Whether to change the voice(pitch) or not
- `noise` : How much noise to add
- `reverb` : Whether to apply reverb

~~~python
def voice_changer(audio_path, pitch, pitch_var, tempo, change, noise, reverb):
    audio, sr = librosa.load(audio_path, sr=22050, dtype='float64')
    ir, sr = librosa.load('/content/00x00y.wav', sr = sr)
    audio = std_change(audio, pitch_var) # variation control
    audio = key_change(audio, pitch) # pitch control
    if change :
        audio = speed_sox(audio, tempo) # tempo(with pitch) control
    else : 
        audio = tempo_sox(audio, tempo) # tempo(without pitch) control
    audio = add_noise(audio, noise) # noise control
    if reverb:
        audio = apply_reverb(audio, ir) # reverb control
    return audio

out = voice_changer('/content/speech.wav', 1.2, 0.5, 0.7, False, 0.01, False)
sf.write('output.wav', out, sr)
~~~

I set the basic sample rate to 44100, but it didn't deform properly, so I changed it to 22050, and I could check the proper sound. (but why?)

And in the case of `speed_sox` and `tempo_sox`, the existing function only generates the file, so we modified it a little bit and made the return value as shown below. As said before, all we have to do is lift the comments on out.

```python
import sox

def speed_sox(audio, rate):
  tfm = sox.Transformer()
  tfm.speed(rate)
  out = tfm.build_array(input_array=audio, sample_rate_in=sr) # Create a return value using 'build_array' instead of 'build_file'
  return out
  # tfm.build_file(output_filepath= 'speech_speed_' + (str)(rate) + '.wav', input_array=audio, sample_rate_in=sr)


def tempo_sox(audio, rate):
  tfm = sox.Transformer()
  tfm.tempo(rate)
  out = tfm.build_array(input_array=audio, sample_rate_in=sr)
  return out
  # tfm.build_file(output_filepath= 'speech_tempo_' + (str)(rate) + '.wav', input_array=audio, sample_rate_in=sr)
```

Lastly, every time I added an effect to the sound source, the size of the sound source itself changed and I checked the problem of clogging the noise operation. Therefore, the internal size variable of `np.random.normal` was changed fluidly.

```python
def add_noise(audio, rate):
    noise = np.random.normal(0,1, np.size(audio)) 
    return audio + rate*noise

sf.write('speech_noise_0.01.wav', add_noise(audio, 0.01), sr)
sf.write('speech_noise_0.1.wav', add_noise(audio, 0.1), sr)
```



We have created a module that identifies the various cognitive elements that make up a given sound source and performs modifications on those that can be controlled. Apart from the elements covered in the process, there is a variety of information that implements linguistic features, tones, more complex emotions, and nonverbal representations, which can be understood and controlled through more complex and sophisticated modeling.

In some cases, the problem was not solved by referring to the document, or hung on to the problem for a few days because I didn't know why. However it was not that difficult. It could be solved by just one or two levels of application and thinking. 

Eventually, I was the first and only one who could complete this content among FutureSkill platform users. I was interested in audio processing before, but I didn't know how to approach it, but I learned a lot in the process of solving problems one by one and referring to various documents. It was a great experience. I want to say thank you to creator Lee Ju-heon for providing precious contents.





---

#### Reference

- [Future Skill](https://futureskill.io/) , Creator Lee Ju-heon

#### Editor

- [Colab](https://colab.research.google.com/)

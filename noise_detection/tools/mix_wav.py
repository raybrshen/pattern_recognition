__author__ = 'ray'

import wave
import numpy as np


wav_1_path = "origin.wav"
wav_2_path = "clap.wav"
wav_out_path = "mixed.wav"

wav_1 = wave.open(wav_1_path, 'rb')
wav_2 = wave.open(wav_2_path, 'rb')
wav_out = wave.open(wav_out_path, 'wb')

len_1 = wav_1.getnframes()
len_2 = wav_2.getnframes()

if len_1>len_2:
    wav_out.setparams(wav_1.getparams())
else:
    wav_out.setparams(wav_2.getparams())

signal_1 = np.fromstring(wav_1.readframes(-1), 'Int16')
signal_2 = np.fromstring(wav_2.readframes(-1), 'Int16')

if len_1>len_2:
    signal_out = np.append(signal_1[:len_2]+signal_2, signal_1[len_2:]).tostring()
elif len_2>len_1:
    signal_out = np.append(signal_1+signal_2[:len_1], signal_2[len_1:]).tostring()
else:
    signal_out = (signal_1+signal_2).tostring()

wav_out.writeframes(signal_out)

wav_1.close()
wav_2.close()
wav_out.close()

print 'done!'

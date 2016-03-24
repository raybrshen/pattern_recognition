__author__ = 'ray'


import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
import os

label_color = {0:'g', 1:'b', 2:'r', 3:'y',4:'c',5:'m'}
audio_path = 'frame_based-clap_cough_speech_env/audios/test/clap_speech_04.wav'
result_path = 'frame_based-clap_cough_speech_env/result_files/clap_speech_04.label'

#---------------------------------- processing audio file ----------------------------------------------------
# open audio file
spf = wave.open(audio_path,'r')
# extract raw audio from wav file
signal = spf.readframes(-1)
# get audio information
sw = spf.getsampwidth()
print 'sample width: ', sw
nc = spf.getnchannels()
print 'number of channels: ', nc
fr = spf.getframerate()
print 'frame rate', fr
nf = spf.getnframes()
print 'number of frames: ', nf
# extract signal value from bytes
signal = np.fromstring(signal, 'Int16')
print 'length of signal: ', len(signal)

# check number of channel
if  nc> 1:
    print 'only support mono audio'
    sys.exit(0)


#---------------------------------- processing result file ----------------------------------------------------
# define lists for packing data
pmat = np.zeros((nf,4))

# open result file
# for align mid and step 30
with open(result_path) as f:
    contents = f.readlines()
    contents = contents[1:]
    left = -661
    right = 661
    for idx, line in enumerate(contents):
        #print line
        label,p3,p2,p1,p0 = line.split(' ')
        #pmat[idx*441:(idx+3)*441-1,:] = pmat[idx*441:(idx+3)*441-1,:] + np.array([float(p0),float(p1),float(p2),float(p3)])
        tmp_left=0 if left<0 else left
        tmp_right=nf if right>nf else right
        pmat[tmp_left:tmp_right+1,:] = pmat[tmp_left:tmp_right+1,:] + np.array([float(p0),float(p1),float(p2),float(p3)])
        left += 441
        right += 441

labels = np.argmax(pmat, 1)
label_time_slots = [{},{},{},{},{},{}]
st_frame = 0
ed_frame = 0
for idx,label in enumerate(labels[1:]):
    idx += 1
    if label==labels[idx-1]:
        ed_frame = idx
    else:
        label_time_slots[labels[idx-1]][st_frame] = ed_frame
        st_frame = ed_frame = idx
ed_frame = nf-1
label_time_slots[labels[-1]][st_frame] = ed_frame

'''
print label_time_slots[0]
print label_time_slots[1]
print label_time_slots[2]
print label_time_slots[3]
'''

#---------------------------------- plot the audio with labels ----------------------------------------------------
# make the x axis in unit millisecond
time=np.linspace(0, len(signal)*1000/float(fr), num=len(signal))
# plot each time slot with corresponding color
plt.figure(1)
plt.title('Signal Wave : '+os.path.basename(audio_path))
#plt.plot(time[0:60000], signal[0:60000], 'r', time[60001:120000], signal[60001:120000], 'b', time[120001:172799], signal[120001:172799], 'g')
#plt.plot(time[0:60000], signal[0:60000], 'r')
#plt.plot(time[60001:120000], signal[60001:120000], 'b')
#plt.plot(time[120001:172799], signal[120001:172799], 'g')
for idx,item in enumerate(label_time_slots[0:4]):
    print 'label: ', idx
    print 'time slots: ', item
    for st,ed in item.iteritems():
        plt.plot(time[st:ed+1], signal[st:ed+1], label_color[idx])
#plt.show()
#---------------------------------- plot the possibility line chart ----------------------------------------------------
# make the x axis in unit frame index
time=np.linspace(0, len(signal)*1000/float(fr), num=len(signal))
# plot the possibility of each class in terms of time
plt.figure(2)
plt.title('Possibility : '+os.path.basename(audio_path))
plt.ylim(-0.1,3.1)
plt.plot(time, pmat[:,0], label_color[0]) # clapping
plt.plot(time, pmat[:,1], label_color[1]) # clapping
plt.plot(time, pmat[:,2], label_color[2]) # environment
plt.plot(time, pmat[:,3], label_color[3]) # speech
#plt.plot(time, p4s, label_color[4]) # laugh
#plt.plot(time, p5s, label_color[5]) # sneeze
plt.show()
'''
'''

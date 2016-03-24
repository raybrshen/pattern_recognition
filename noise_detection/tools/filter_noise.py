__author__ = 'ray'

import os
from subprocess import Popen,PIPE
import wave
import numpy as np

def run_comm(comm):
    process = Popen(comm, stdout=PIPE)
    (std_out,std_err) = process.communicate()
    exit_code = process.wait()
    if std_out: print std_out
    if std_err!=None: print std_err

def run_comm_save_output(comm, out_file):
    process = Popen(comm, stdout=PIPE)
    (std_out,std_err) = process.communicate()
    exit_code = process.wait()
    of = open(out_file, 'w')
    of.write(std_out)
    of.close()
    if std_err!=None: print std_err

in_wav_path = 'frame_based-clap_cough_speech_env/test/test_clap'
out_wav_path = 'frame_based-clap_cough_speech_env/test/test_clap_filter.wav'

smile_exe = 'frame_based-clap_cough_speech_env/tools/SMILExtract'
config_file = 'frame_based-clap_cough_speech_env/tools/step_30_align_mid_without_energy.conf'
scale_exe = 'frame_based-clap_cough_speech_env/tools/svm-scale'
range_file = 'frame_based-clap_cough_speech_env/model/training_set.range'
predict_exe = 'frame_based-clap_cough_speech_env/tools/svm-predict'
model_file = 'frame_based-clap_cough_speech_env/model/training_set.model'

# extract features with openSMILE
assert os.path.isfile(in_wav_path+'.wav')
comm = [smile_exe,'-C',config_file,'-I',in_wav_path+'.wav',"-O",in_wav_path+'.rawdata']
run_comm(comm)

# scale the features
assert os.path.isfile(in_wav_path+'.rawdata')
comm = [scale_exe, '-r', range_file, in_wav_path+'.rawdata']
run_comm_save_output(comm, in_wav_path+".scale")
os.remove(in_wav_path+'.rawdata')

# predict result from features
assert os.path.isfile(in_wav_path+'.scale')
comm = [predict_exe, '-b', '1', in_wav_path+'.scale', model_file, in_wav_path+'.label']
run_comm(comm)
os.remove(in_wav_path+'.scale')


# open audio file
in_wav = wave.open(in_wav_path+'.wav','rb')
nf = in_wav.getnframes()
# set parameters of the output wave the same as input wave
out_wav = wave.open(out_wav_path, 'wb')
out_wav.setparams(in_wav.getparams())

# process label file to get noise frames
assert os.path.isfile(in_wav_path+'.label')
# define lists for packing data
pmat = np.zeros((nf,4))
with open(in_wav_path+'.label') as f:
    contents = f.readlines()
    contents = contents[1:]
    # start the possibility overlapping process
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
os.remove(in_wav_path+'.label')


# get label of each frame
labels = np.argmax(pmat, 1)
# extract signal value from bytes
in_signal = np.fromstring(in_wav.readframes(-1), 'Int16')
# assert consistency of frame number
assert len(in_signal)==len(labels)
# write to output wave only non-noise frames
out_wav.writeframes(in_signal[labels!=2].tostring())

# close files
in_wav.close()
out_wav.close()

assert os.path.isfile(out_wav_path)
print 'done!'

__author__ = 'ray'

import wave
import random
import os
import numpy as np

root_path = 'asr/'
wav_scp_file = root_path+'wav.scp'
text_file = root_path+'text'

noise_types = ['clap', 'cough']
noise_tag_list = ['<NOISE>', '<SPOKEN_NOISE>']



def add_sound(in_file, sound_file, out_file, is_prefix):
    wav_in = wave.open(in_file, 'rb')
    wav_sound = wave.open(sound_file, 'rb')
    wav_out = wave.open(out_file, 'wb')
    assert wav_in.getframerate() == wav_sound.getframerate()

    len_in = wav_in.getnframes()
    len_sound = wav_sound.getnframes()

    wav_out.setparams(wav_in.getparams())
    wav_out.setnframes(len_in+len_sound)

    if is_prefix:
        wav_out.writeframes(wav_sound.readframes(-1))
        wav_out.writeframes(wav_in.readframes(-1))
    else:
        wav_out.writeframes(wav_in.readframes(-1))
        wav_out.writeframes(wav_sound.readframes(-1))

    wav_in.close()
    wav_sound.close()
    wav_out.close()


if __name__ == '__main__':
    # read the whole wav.scp into a list
    with open(wav_scp_file) as f:
        lines_wav_scp = f.readlines()
        num_audio = len(lines_wav_scp)
        wave_paths = [lines_wav_scp[i].strip().split(' ')[1] for i in range(0,num_audio)]

    with open(text_file) as f:
        lines_text = f.readlines()
        assert len(lines_text)==num_audio

    # get a list of random index for each noise file
    num_noise = 0
    for root, dirs, files in os.walk('asr/noise/'):
        for file in files:
            if file.endswith('.wav'):
                num_noise += 1

    # index for lopping random audio files from the corpus
    idx = 0
    # each noise file has two modes: prefix and suffix
    random_indexes = random.sample(range(0,num_audio),2*num_noise)

    # loop all kinds of noise
    for i,noise_type in enumerate(noise_types):
        # loop all the directories for this type of noise
        noise_path = root_path+'noise/'+noise_type+'/'
        sub_paths = os.listdir(noise_path)
        for sub_path in sub_paths:
            noise_sub_path = noise_path+sub_path+'/'
            noise_files = os.listdir(noise_sub_path)
            log_file = root_path+'logs/'+noise_type+'-'+sub_path+'.log'
            if os.path.exists(log_file):
                os.remove(log_file)
            # loop all the clapping audio inside the directory
            for noise_file in noise_files:
                #-- prefix the noise audio --
                r_idx = random_indexes[idx]
                idx += 1
                # fix text
                content_split = lines_text[r_idx].strip().split(' ')
                content_split.insert(1,' '.join([noise_tag_list[i]]*int(sub_path)))
                lines_text[r_idx] = ' '.join(content_split)+'\n'
                # fix wav.scp
                line = lines_wav_scp[r_idx].strip()
                lines_wav_scp[r_idx] = line[0:-4]+'_noise.wav\n'
                # get audio path from wav.scp
                in_file = root_path+wave_paths[r_idx]
                # add noise to audio
                out_file = in_file[0:-4]+'_noise.wav'
                add_sound(in_file,noise_sub_path+noise_file,out_file,True)
                # log information
                with open(log_file,'a') as logfile:
                    logfile.write(out_file+'\n')
                #-- suffix the noise audio --
                r_idx = random_indexes[idx]
                idx += 1
                # fix text
                content_split = lines_text[r_idx].strip().split(' ')
                content_split.append(' '.join([noise_tag_list[i]]*int(sub_path)))
                lines_text[r_idx] = ' '.join(content_split)+'\n'
                # fix wav.scp
                line = lines_wav_scp[r_idx].strip()
                lines_wav_scp[r_idx] = line[0:-4]+'_noise.wav\n'
                # get audio path from wav.scp
                in_file = root_path+wave_paths[r_idx]
                # add noise to audio
                out_file = in_file[0:-4]+'_noise.wav'
                add_sound(in_file,noise_sub_path+noise_file,out_file,False)
                # log information
                with open(log_file,'a') as logfile:
                    logfile.write(out_file+'\n')

    with open(text_file+'.new','w') as new_text:
        new_text.writelines(lines_text)
    with open(wav_scp_file+'.new','w') as new_wav_scp:
        new_wav_scp.writelines(lines_wav_scp)

    print 'done!'



__author__ = 'ray'

import os
from subprocess import Popen,PIPE


source_path = "frame_based-clap_cough_speech_env/audios"
destination_path = "frame_based-clap_cough_speech_env/raw_data"
smile_exe = "frame_based-clap_cough_speech_env/tools/SMILExtract"
smile_conf = "frame_based-clap_cough_speech_env/tools/step_25_align_left.conf"

for root, dirs, files in os.walk(source_path):
    for file in files:
        if file.endswith(".wav"):
            input_file = root+"/"+file
            output_file = destination_path+"/"+file.replace('.wav','.rawdata')
            '''
            #print input_file
            #print output_file
            '''
            command = [smile_exe,"-C",smile_conf,"-I",input_file,"-O",output_file]
            process = Popen(command, stdout=PIPE)
            (output,err) = process.communicate()
            exit_code = process.wait()
            if output != '':
                print output
            if err != None:
                print err

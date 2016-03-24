__author__ = 'ray'


import re
import sys
import fileinput
import operator


label_index = {'env':'0', 'speech':'1', 'clap':'2', 'cough':'3', 'laugh':'4', 'sneeze':'5'}
index_number = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5}
number_index = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5'}

def label_sound(in_file_name, out_file_name, label, ranges):

    out_file = open(out_file_name, 'a')

    with open(in_file_name) as f:
        lines = f.readlines()
    #print len(lines)

    last_end = -1
    for start,end in ranges.iteritems():
        if start <= last_end:
            print "error: start time <= last end time : %d<%d" % (start,last_end)
            exit()
        if end < start:
            print "error: end time < start time : %d<%d" % (end,start)
            exit()
        for i in range(start,end+1):
            #print label+lines[i][1:]
            out_file.write(label+lines[i][1:])

    out_file.close()


if __name__ == '__main__':
    #if len(sys.argv)==1: print sys.argv

    training_dir = 'frame_based-clap_cough_speech_env/raw_data/training/'

    # env: 1101 frames
    # speech: 3340 frames
    # clap: 529 frames (454 without low-volume examples)
    # cough: 938 frames (492 without artificial examples)

    # long_env.wav
    ranges = {0:1100}
    in_file_name = training_dir+"env.rawdata"
    out_file_name = training_dir+"training_set.rawdata"
    label_sound(in_file_name, out_file_name, label_index["env"], ranges)

    print "done!"

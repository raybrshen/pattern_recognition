__author__ = 'ray'

import re
import sys
import fileinput
import operator

label_index = {'unknown':'0', 'speech':'1', 'clapping':'2', 'stepping':'3', 'flipping':'4'}
index_number = {'0':0, '1':1, '2':2, '3':3, '4':4}
number_index = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4'}

def process_text_file(in_file_name, out_file_name):

    # open files
    in_file = open(in_file_name, 'r')
    out_file = open(out_file_name, 'w')

    # regular expression for selecting data
    #pattern = re.compile("^\'[a-zA-Z]+_[0-9]+\'$")
    pattern = re.compile("^\'[a-zA-Z0-9]+\'$")

    for line in in_file:
        contents = line.split(',')
        contents_size = len(contents)
        if contents_size==6555 and pattern.match(contents[0]):
            for i in range(1,contents_size-1):
                contents[i] = str(i)+':'+contents[i]
            #print labels[contents[contents_size-1].strip()]+' '+' '.join(contents[1:contents_size-1])
            out_file.write(label_index[contents[contents_size-1].strip()]+' '+' '.join(contents[1:contents_size-1])+'\n')

    # close files
    in_file.close()
    out_file.close()


def remove_label(file_name):
    # the inplace parameter will redirect stdout to the file, so that the print function will write back to the file
    for line in fileinput.input(file_name, inplace=1):
        contents = line.split(' ')
        contents_size = len(contents)
        #print contents_size
        if contents_size>1:
            sys.stdout.write(str(0)+' '+' '.join(contents[1:]))


def shift_order(in_file_name, out_file_name):

    # open files
    in_file = open(in_file_name, 'r')
    out_file = open(out_file_name, 'w')

    datalist = []
    for line in in_file:
        contents = line.split(' ', 1)
        datalist.append((index_number[contents[0]],contents[1]))

    sorted_list = sorted(datalist, key=operator.itemgetter(0))

    for listitem in sorted_list:
        out_file.write(number_index[listitem[0]]+' '+listitem[1])

    # close files
    in_file.close()
    out_file.close()


def evaluate_re(str):
    print str
    #pattern = re.compile("^\'[a-zA-Z]+_[0-9]+\'$")
    pattern = re.compile("^\'[a-zA-Z0-9]+\'$")

    if pattern.match(str):
        print "true"
    else:
        print "false"


def display(in_file_name):
    #pattern = re.compile("^\'[a-zA-Z]+_[0-9]+\'$")
    pattern = re.compile("^\'[a-zA-Z0-9]+\'$")

    in_file = open(in_file_name, 'r')
    #out_file = open(out_file_name, 'w')

    for line in in_file:
        contents = line.split(',')
        contents_size = len(contents)
        if contents_size==6555 and pattern.match(contents[0]):
            print contents[0]


if __name__ == '__main__':
    if len(sys.argv)==1:
        print sys.argv
    #display("sound.arff")
    #evaluate_re("\'stepping_16\'")
    #process_text_file("sound.arff", "sound.rawdata")
    #remove_label("testing_set-shift_order.scale")
    #shift_order("testing_set.scale", "testing_set-shift_order.scale")

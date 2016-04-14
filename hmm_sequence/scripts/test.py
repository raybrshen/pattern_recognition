#
# created by ray on 2014-04-08
#

import numpy as np

# train_file_1 = '../data/without_punctuation/only_tag_train_declarative.txt'
# train_file_2 = '../data/without_punctuation/only_tag_train_imperative.txt'
# train_file_3 = '../data/without_punctuation/only_tag_train_interrogative.txt'
classes = ['declarative','imperative','interrogative']

for i in xrange(1,101):
  for cls in classes:
    tfile = '../data/without_punctuation/only_tag_train_'+cls+'.txt'
    with open(tfile,'r') as fr:
      lines = []
      for j in xrange(i): lines.append(fr.readline())
    wfile = '../data/tmp/'+cls+'_'+str(i)+'.txt'
    with open(wfile,'w') as fw:
      fw.writelines(lines)

#
# if __name__ == '__main__':
#   hmm_learn()




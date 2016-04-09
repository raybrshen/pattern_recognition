#
# created by ray on 2014-04-08
#

import os, nltk
from nltk.corpus import wordnet as wn



TAG_SET_PATH = 'help/tagsets/upenn_tagset.pickle'
ROOT_PATH = '../Data/sentence/'


def pos_tag(text):
  # print text
  tokens = nltk.word_tokenize(text)
  # print tokens
  tags = nltk.pos_tag(tokens)
  # print tags
  # print [t[1] for t in tags]
  return [t[1] for t in tags]

def list_files(path, startswith=''):
  files = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if startswith and not filename.startswith(startswith): continue
      files.append(dirpath+'/'+filename)
  return files

def wordnet_tmp():
  word = 'see'
  print(wn.synset(word+'.v.01').definition())
  return

def sentence2tag():
  files = list_files(ROOT_PATH + 'origin')
  f_train = open(ROOT_PATH + 'without_punctuation/train_tag.txt', 'w')
  f_test = open(ROOT_PATH + 'without_punctuation/test_tag.txt', 'w')
  f_train_text = open(ROOT_PATH + 'without_punctuation/train_text.txt', 'w')
  f_test_text = open(ROOT_PATH + 'without_punctuation/test_text.txt', 'w')
  for idx, filename in enumerate(files):
    print filename
    fr = open(filename, 'r')
    lines = fr.readlines()
    fr.close()
    # fw = open(path+'/tag_'+os.path.basename(filename), 'w')
    i = 0
    for line in lines:
      tags = [tag for tag in pos_tag(line) if tag[0].isalpha()]
      tagline = ' '.join(tags) + ' ' + str(idx)
      if i < 100:
        f_train.write(tagline + '\n'); f_train_text.write(line)
      else:
        f_test.write(tagline + '\n'); f_test_text.write(line)
      i += 1
      # fw.close()
  f_train.close()
  f_test.close()
  f_train_text.close()
  f_test_text.close()
  return

def append_epsilons():
  read_path = ROOT_PATH + 'without_punctuation/'
  write_path = ROOT_PATH
  files = ('train_tag.txt', 'test_tag.txt')
  for fn in files:
    print read_path+fn
    with open(read_path+fn, 'r') as fr: lines=fr.readlines()
    with open(write_path+fn, 'w') as fw:
      for line in lines: fw.write('EPS EPS '+line)
    return

def get_tag2id():
  tag_dict = nltk.data.load(TAG_SET_PATH)
  tags = ['EPS'] + [ tag for tag in tag_dict.iterkeys() if tag[0].isalpha() ]
  tag2id = { tag:idx for idx,tag in enumerate(tags) }
  return tag2id




if __name__ == '__main__':
  _t2i = get_tag2id()
  print len(_t2i)
  for k,v in _t2i.iteritems(): print k,v

# if __name__ == '__main__':
  # pos_tag('hello word, i went to the library today.')
  # pos_tag('how are you today')
  # pos_tag('what is the temperature in Seattle now')
  # pos_tag('raise up your hand')
  # pos_tag('Be there at 5:00')

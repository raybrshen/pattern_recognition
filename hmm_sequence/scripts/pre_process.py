#
# created by ray on 2014-04-08
#

import os, nltk
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.tag.perceptron import PerceptronTagger


TAG_SET_PATH = 'help/tagsets/upenn_tagset.pickle'
ROOT_PATH = '../data/'


def pos_tag(text):
  # print text
  tags = tagger.tag(word_tokenize(text))
  # print tags
  return [t[1] for t in tags]

def list_files(path, startswith=''):
  ret = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if startswith and not filename.startswith(startswith): continue
      ret.append(dirpath+'/'+filename)
  return ret

def wordnet_tmp():
  word = 'see'
  print(wn.synset(word+'.v.01').definition())
  return

def sentence2tags(in_file, out_file):
  fin,fout = open(in_file,'r'),open(out_file, 'w')
  lines = fin.readlines()
  for line in lines:
    tags = [tag for tag in pos_tag(line) if tag[0].isalpha()]
    fout.write(' '.join(tags)+'\n')
  fin.close()
  fout.close()
  return

def get_tag2id():
  tag_dict = nltk.data.load(TAG_SET_PATH)
  tags = ['EPS'] + [ tag for tag in tag_dict.iterkeys() if tag[0].isalpha() ]
  tag2id = { tag:idx for idx,tag in enumerate(tags) }
  return tag2id




if __name__ == '__main__':
  tagger = PerceptronTagger()
  files = list_files(ROOT_PATH+'origin')
  for fn in files:
    print fn
    _dir,_base = os.path.dirname(fn),os.path.basename(fn)
    sentence2tags(fn,_dir+'/tag_'+_base)


# if __name__ == '__main__':
  # pos_tag('hello word, i went to the library today.')
  # pos_tag('how are you today')
  # pos_tag('what is the temperature in Seattle now')
  # pos_tag('raise up your hand')
  # pos_tag('Be there at 5:00')

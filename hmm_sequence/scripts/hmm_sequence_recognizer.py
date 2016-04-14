# 
# created by ray on 2016-04-09
#

import pickle
from nltk import word_tokenize
from nltk.tag.perceptron import PerceptronTagger
from hmm_model import HmmModel
import matplotlib.pyplot as plt



class HmmSeqRecognizer(object):
  def __init__(self, _tagger):
    self.hmm_models = []
    self.n_hmm = 0
    self.hmm2idx = {}
    self.idx2hmm = {}
    # print '=> loading tagger...'
    self.tagger = _tagger
    # print '|  done'
    return

  def cross_validation(self, samples, label):
    tp,ns = 0,len(samples)
    for i in xrange(ns):
      idx = self.predict_sample(samples[i])
      if idx==label: tp+=1
    return tp,float(tp)/ns

  def predict_sample(self, sample):
    sample = [sample]
    probs = [ model.test(sample) for model in self.hmm_models ]
    return probs.index(max(probs))

  def predict_sentence(self, sentence):
    sample =  [[ tag for _,tag in self.tagger.tag(word_tokenize(sentence)) ]]
    probs = [ model.test(sample) for model in self.hmm_models ]
    return probs.index(max(probs))

  def add_model(self, name, model):
    self.hmm_models.append(model)
    self.hmm2idx[name] = self.n_hmm
    self.idx2hmm[self.n_hmm] = name
    self.n_hmm += 1

  def new_hmm(self, name, datapath, nhs, ne):
    # print '=> adding HMM model \'%s\'...' % name
    hmm_model = HmmModel(nhs)
    hmm_model.train(datapath,ne)
    self.add_model(name, hmm_model)
    # print '|  done'
    return

  def save_hmm(self, name, hmm_path):
    print '=> saving HMM model \'%s\'...' % name
    f = open(hmm_path, 'wb')
    pickle.dump(self.hmm_models[self.hmm2idx[name]], f)
    f.close()
    print '|  done'
    return

  def load_hmm(self, name, hmm_path):
    print '=> adding HMM model \'%s\'...' % name
    f = open(hmm_path, 'rb')
    hmm_model = pickle.load(f)
    f.close()
    self.add_model(name, hmm_model)
    print '|  done'
    return




def random_test():
  # testing random sentences
  print 'number of hmm: %d' % hmm_recognizer.n_hmm
  test_set = ['what day is it today',
              'i have an apple',
              'please raise up your hand']
  for sentence in test_set:
    idx = hmm_recognizer.predict_sentence(sentence)
    name = hmm_recognizer.idx2hmm[idx]
    print 'predicting: %s' % sentence
    print 'result: %s(%d)' % (name, idx)


# global test function
def batch_test():
  tp_all,n_all = 0,0
  _data_path = '../data/without_punctuation/'
  ret = []
  for cls in classes:
    # print '=> testing \'%s\' samples...' % cls
    datafile = _data_path + 'only_tag_test_' + cls + '.txt'
    idx = hmm_recognizer.hmm2idx[cls]
    model = hmm_recognizer.hmm_models[idx]
    samples = model.load_samples(datafile, False)
    n_all += len(samples)
    tp,acc = hmm_recognizer.cross_validation(samples,idx)
    ret.append(acc)
    tp_all += tp
    # print '|  true positives: %s, accuracy: %g' % (tp,acc)
  # print '=> done (overall accuracy:%g)' % (float(tp_all)/n_all)
  return ret



if __name__ == '__main__':

  ret = []
  tagger = PerceptronTagger()
  ne,nhs = 1,5
  classes = ['declarative', 'imperative', 'interrogative']
  data_path = '../data/tmp/'


  acc1,acc2,acc3 = [],[],[]

  for i in xrange(1,101):
    hmm_recognizer = HmmSeqRecognizer(tagger)
    for cls in classes:
      # training new HMM models and then save to files
      data_file = data_path + cls +'_'+str(i)+ '.txt'
      hmm_recognizer.new_hmm(cls, data_file, nhs, ne)

    print '\r    '+str(i),
    acc = batch_test()
    acc1.append(acc[0])
    acc2.append(acc[1])
    acc3.append(acc[2])
    # print 'epoch: %d, acc: %g,%g,%g' % (i, acc[0],acc[1],acc[2])
  plt.plot(range(1,101), acc1)
  plt.show()
  plt.plot(range(1, 101), acc2)
  plt.show()
  plt.plot(range(1, 101), acc3)
  plt.show()
  # print ret












# if __name__ == '__main__':
#   hmm_recognizer = HmmSeqRecognizer()
#
#   # configuration
#   ne,nhs = 1,5
#   classes = ['declarative', 'imperative', 'interrogative']
#   data_path = '../data/without_punctuation/'
#   model_path = '../data/models/'
#
#   # training new model or loading existing model
#   train_flag = False
#   for cls in classes:
#     # get model file path as configured
#     model_file = model_path + cls + '_%dhs%depoch.model' % (nhs, ne)
#     if train_flag:
#       # training new HMM models and then save to files
#       data_file = data_path + 'only_tag_train_' + cls + '.txt'
#       hmm_recognizer.new_hmm(cls, data_file, nhs, ne)
#       hmm_recognizer.save_hmm(cls, model_file)
#     else:
#       # loading HMM models from files
#       hmm_recognizer.load_hmm(cls, model_file)
#
#   # random_test()
#   batch_test()







# def train_converge():
#   from hmmlearn import hmm
#   # model = hmm.MultinomialHMM(n_components=5)
#   nhs = 5
#   train_file_1 = '../data/without_punctuation/declarative/only_tag_train_declarative.txt'
#   train_file_2 = '../data/without_punctuation/declarative/only_tag_train_imperative.txt'
#   train_file_3 = '../data/without_punctuation/declarative/only_tag_train_interrogative.txt'
#   model_1,model_2,model_3 = HmmModel(nhs),HmmModel(nhs),HmmModel(nhs)
#   model_1.model = hmm.MultinomialHMM(n_components=nhs)
#   model_2.model = hmm.MultinomialHMM(n_components=nhs)
#   model_3.model = hmm.MultinomialHMM(n_components=nhs)
#   train_samples_1 = []
#   train_samples_2 = model_1.load_samples(train_file_2, True)
#   train_samples_3 = model_1.load_samples(train_file_3, True)
#   recog = HmmSeqRecognizer()
#   recog.add_model('declarative',model_1)
#   recog.add_model('imperative',model_2)
#   recog.add_model('interrogative',model_3)
#   for i in xrange(100):
#     sa_1,l_1 = model_1.samples2array(train_samples_1[i])
#     sa_2,l_2 = model_2.samples2array(train_samples_2[i])
#     sa_3,l_3 = model_3.samples2array(train_samples_3[i])
#     model_1.model.fit(sa_1,l_1)
#     model_1.model.fit(sa_2,l_2)
#     model_1.model.fit(sa_3,l_3)
#
#
#   for cls in classes:
#     datafile = data_path + 'only_tag_test_' + cls + '.txt'
#     with open(datafile) as df:
#       lines = df.readlines()
#       samples = []
#       for line in lines: samples.append(line.split())
#!/usr/bin/python
# 
# created by ray on 2016-04-25
# 

from nltk.tag.perceptron import PerceptronTagger
from pre_process import list_files
from hmm_model import HmmModel
import matplotlib.pyplot as plt


def load_data(_files):
  _data = []
  for _fn in _files:
    with open(_fn, 'r') as _f:
      _lines = _f.readlines()
    _d = [ _line.strip().split() for _line in _lines ]
    _data.append(_d)
  return _data

def separate_data(_data, _idx, _fold):
  _train,_test = [],[]
  for _d in _data:
    _size = len(_d)/_fold
    _test.append(_d[(_idx-1)*_size:_idx*_size])
    _train.append(_d[:(_idx-1)*_size] + _d[_idx*_size:])
  return _train,_test

def training(_models,_train):
  for _i in xrange(len(_models)):
    _samples = _models[_i].load_samples_from_arr(_train[_i],True)
    _models[_i].train_from_samples(_samples)
  return

def predict(_models,_tags):
  _probs = []
  for _model in _models:
    _probs.append(_model.test([_tags]))
  _cls = _probs.index(max(_probs))
  return _cls

def testing(_models,_test,_confusion):
  _tp,_n = 0,0
  for _i in xrange(len(_test)):
    _n += len(_test[_i])
    for _tags in _test[_i]:
      _cls = predict(_models,_tags)
      _confusion[_i][_cls] += 1
      if _cls==_i: _tp+=1
  return float(_tp)/_n

def analyze(_confusion):
  _n = len(_confusion)
  _total,_tp = 0,0
  for _i in xrange(_n):
    if not _i: print '\t\tcls-%d'%(_i+1),
    else: print '\tcls-%d'%(_i+1),
  print ''
  for _i in xrange(_n):
    print 'cls-%d\t'%(_i+1),
    for _j in xrange(_n):
      _tmp = _confusion[_i][_j]
      _total += _tmp
      if _i==_j: _tp+= _tmp
      print '%d\t\t'%_tmp,
    print ''
  return float(_tp)/_total

def cross_validation(nhs,fold):
  files = list_files('../data/origin','tag_')
  print files
  data = load_data(files)
  models = [ HmmModel(nhs) for _ in xrange(3) ]
  # classes = {'declarative':1,'imperative':2,'interrogative':3}
  confusion = [ [0]*3 for _ in xrange(3) ]
  print 'cross-validation...'
  for i in xrange(1,fold+1):
    train,test = separate_data(data,i,fold)
    training(models,train)
    acc_i = testing(models,test,confusion)
    print '%d-fold testing accuracy: %g' % (i,acc_i)
  acc = analyze(confusion)
  print '%d-nhs overall accuracy: %g' % (nhs,acc)
  return acc

if __name__ == '__main__':
  # print 'loading tagger...'
  # tagger = PerceptronTagger()
  # x = range(2, 21)
  # y = []
  # for i in xrange(2,21):
  #   y.append(cross_validation(i,10))
  #   print ''
  # print y
  x = range(2,11)
  y = [0.713333, 0.751111, 0.777778, 0.757778, 0.773333, 0.742222, 0.775556, 0.773333, 0.757778]
  plt.plot(x, y)
  plt.show()

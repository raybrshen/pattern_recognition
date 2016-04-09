#
# created by ray on 2014-04-08
#

import numpy as np


def extract(fn):
  samples = []
  lengths = []
  sample_array = []
  with open(fn,'r') as fr:
    lines = fr.readlines()
  t2i,i2t = {},{}
  for line in lines:
    # sample = line.split()
    sample = line.split()
    for tag in sample:
      if tag not in t2i:
        t2i[tag] = len(t2i)
        i2t[len(t2i)-1] = tag
    samples.append(sample)
    lengths.append(len(sample))
  for sample in samples:
    for i in xrange(len(sample)):
      # sample[i] = [tag_set[sample[i]]]
      sample_array += [[t2i[sample[i]]]]
  return t2i,i2t,lengths,samples,sample_array

def hmm_learn():
  from hmmlearn import hmm
  t2i_1,i2t_1,l_1,s_1,sa_1 = extract('../Data/sentence/without_punctuation/only_tag_train_declarative.txt')
  # uniq = set([ tag[0] for tag in sa ])
  # print len(uniq), uniq
  t2i_2,i2t_2,l_2,s_2,sa_2 = extract('../Data/sentence/without_punctuation/only_tag_train_imperative.txt')
  t2i_3,i2t_3,l_3,s_3,sa_3 = extract('../Data/sentence/without_punctuation/only_tag_train_interrogative.txt')

  # transform into numpy format matrix
  sa_1 = np.array(sa_1)
  sa_2 = np.array(sa_2)
  sa_3 = np.array(sa_3)

  nc = 5
  model_1 = hmm.MultinomialHMM(n_components=nc)
  model_2 = hmm.MultinomialHMM(n_components=nc)
  model_3 = hmm.MultinomialHMM(n_components=nc)
  for epoch in xrange(1):
    print 'training epoch %d' % (epoch+1)
    model_1.fit(sa_1, l_1)
    model_2.fit(sa_2, l_2)
    model_3.fit(sa_3, l_3)

  # perform testing on training set
  # _ss = [s_1, s_2, s_3]
  # for idx in xrange(len(_ss)):
  #   tp = 0
  #   ns = len(_ss[idx])
  #   print ns
  #   for i in xrange(ns):
  #     _s = _ss[idx][i]
  #     _s_1 = [ [t2i_1[t]] for t in _s if t in t2i_1 ]
  #     _s_2 = [ [t2i_2[t]] for t in _s if t in t2i_2 ]
  #     _s_3 = [ [t2i_3[t]] for t in _s if t in t2i_3 ]
  #     prob_1,prob_2,prob_3 = model_1.decode(_s_1, [len(_s_1)]),model_2.decode(_s_2, [len(_s_2)]),model_3.decode(_s_3, [len(_s_3)])
  #     probs = [prob_1, prob_2, prob_3]
  #     mini,maxi = probs.index(min(probs))+1,probs.index(max(probs))+1
  #     # print 'min: %d, max: %d\n' % (mini,maxi)
  #     if maxi==idx+1: tp+=1
  #   print 'set %d | tp: %d, acc: %g' % (idx+1,tp,float(tp)/ns)

  # perform testing on test set
  _,_,_,s_1,_ = extract('../Data/sentence/without_punctuation/only_tag_test_declarative.txt')
  _,_,_,s_2,_ = extract('../Data/sentence/without_punctuation/only_tag_test_imperative.txt')
  _,_,_,s_3,_ = extract('../Data/sentence/without_punctuation/only_tag_test_interrogative.txt')
  _ss = [s_1, s_2, s_3]
  for idx in xrange(len(_ss)):
    tp = 0
    ns = len(_ss[idx])
    print ns
    for i in xrange(ns):
      _s = _ss[idx][i]
      _s_1 = [ [t2i_1[t]] for t in _s if t in t2i_1 ]
      _s_2 = [ [t2i_2[t]] for t in _s if t in t2i_2 ]
      _s_3 = [ [t2i_3[t]] for t in _s if t in t2i_3 ]
      prob_1,prob_2,prob_3 = model_1.decode(_s_1, [len(_s_1)]),model_2.decode(_s_2, [len(_s_2)]),model_3.decode(_s_3, [len(_s_3)])
      probs = [prob_1, prob_2, prob_3]
      mini,maxi = probs.index(min(probs))+1,probs.index(max(probs))+1
      # print 'min: %d, max: %d\n' % (mini,maxi)
      if maxi==idx+1: tp+=1
    print 'set %d | tp: %d, acc: %g' % (idx+1,tp,float(tp)/ns)

  # print s_1[0]
  # _s,_l = s_1[0],[l_1[0]]
  # prob_1,seq_1 = model_1.decode(_s, _l)
  # prob_2,seq_2 = model_2.decode(_s, _l)
  # prob_3,seq_3 = model_3.decode(_s, _l)
  # print prob_1, seq_1
  # print prob_2, seq_2
  # print prob_3, seq_3
  # probs = [prob_1,prob_2,prob_3]
  # print 'min: %d, max: %d\n' % (probs.index(min(probs))+1,probs.index(max(probs))+1)
  # print s_2[0]
  # _s,_l = s_2[0],[l_2[0]]
  # prob_1,seq_1 = model_1.decode(_s, _l)
  # prob_2,seq_2 = model_2.decode(_s, _l)
  # prob_3,seq_3 = model_3.decode(_s, _l)
  # print prob_1, seq_1
  # print prob_2, seq_2
  # print prob_3, seq_3
  # probs = [prob_1,prob_2,prob_3]
  # print 'min: %d, max: %d\n' % (probs.index(min(probs))+1,probs.index(max(probs))+1)
  # print s_3[0]
  # _s,_l = s_3[0],[l_3[0]]
  # prob_1,seq_1 = model_1.decode(_s, _l)
  # prob_2,seq_2 = model_2.decode(_s, _l)
  # prob_3,seq_3 = model_3.decode(_s, _l)
  # print prob_1, seq_1
  # print prob_2, seq_2
  # print prob_3, seq_3
  # probs = [prob_1,prob_2,prob_3]
  # print 'min: %d, max: %d\n' % (probs.index(min(probs))+1,probs.index(max(probs))+1)
  return




if __name__ == '__main__':
  hmm_learn()




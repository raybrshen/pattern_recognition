# 
# created by ray on 2016-04-09
# 

import numpy as np
from hmmlearn import hmm


class HmmModel(object):
  def __init__(self, nhs):
    self.tag2idx = {}
    self.idx2tag = {}
    self.model = None
    self.n_state = -1
    self.n_hidden_state = nhs
    return

  def load_samples(self, datapath, update=False):
    samples = []
    # reset if required
    if update:
      self.n_state = 0
      self.tag2idx, self.idx2tag = {}, {}
    # load text data from file
    with open(datapath, 'r') as fr:
      lines = fr.readlines()
    # extract samples and samples length
    for line in lines:
      tags = line.split()
      if update:
        # update current model's tag set
        sample = []
        for tag in tags:
          if tag not in self.tag2idx:
            self.tag2idx[tag] = self.n_state
            self.idx2tag[self.n_state] = tag
            self.n_state += 1
          sample.append(tag)
      else:
        # only add tags that exist in current model's tag set
        sample = [ tag for tag in tags if tag in self.tag2idx]
      # add this sample into samples set
      samples.append(sample)
    return samples

  def samples2array(self, samples):
    samples_array,samples_length = [],[]
    # flatten rows of samples to a single column vector
    # used by the training function
    for sample in samples:
      length = 0
      for tag in sample:
        if tag not in self.tag2idx: continue
        samples_array.append([self.tag2idx[tag]])
        length += 1
      if length: samples_length.append(length)
    # return empty array if none of the tags exist in current model's tag set
    return samples_array,samples_length

  def train(self, datapath, n_epoch):
    self.model = hmm.MultinomialHMM(n_components=self.n_hidden_state)
    print '|  loading training data...'
    samples = self.load_samples(datapath, True)
    samples_array,samples_length = self.samples2array(samples)
    print '|  start training'
    for i in xrange(n_epoch):
      print '|  |  training epoch %d...' % (i + 1)
      self.model.fit(samples_array, samples_length)
    print '|  finished training'
    return

  def test(self, sample):
    sa,l = self.samples2array(sample)
    # if none of the tags in the sample exist in current model's tag set
    # return minus infinite to indicate very low possibility
    if not sa: return -float('inf')
    sa = np.array(sa)
    prob,_ = self.model.decode(sa,l)
    return prob



if __name__ == '__main__':
  pass


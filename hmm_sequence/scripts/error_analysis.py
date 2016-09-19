#!/usr/bin/python

data_path = '../data/origin/'
declarative_path = data_path+'tag_declarative.txt'
imperative_path = data_path+'tag_imperative.txt'
interrogative_path = data_path+'tag_interrogative.txt'

n_classes = 3
declarative,imperative,interrogative = 0,1,2
declarative_sents = {}
imperative_sents = {}
interrogative_sents = {}
classes = ['declarative','imperative','interrogative']
files = [declarative_path,imperative_path,interrogative_path]
sents = [declarative_sents,imperative_sents,interrogative_sents]

for i in xrange(n_classes):
  print 'processing: '+files[i]+'...'
  with open(files[i]) as f: contents=f.readlines()
  for line in contents:
    line = line.strip()
    sent = tuple(line.split())
    if sent in sents[i]: sents[i][sent]+=1
    else: sents[i][sent] = 1

print 'number of sents ' + classes[0] + ': '+str(len(sents[0]))
print 'number of sents ' + classes[1] + ': '+str(len(sents[1]))
print 'number of sents ' + classes[2] + ': '+str(len(sents[2]))

c1 = 0 #declarative
c2 = 2 #interrogative
n_same = 0
for sent in sents[c1]:
  if sent not in sents[c2]: continue
  print sent
  print classes[c1]+':'+str(sents[c1][sent])
  print classes[c2]+':'+str(sents[c2][sent])




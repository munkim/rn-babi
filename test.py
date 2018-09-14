import random
import torch
import numpy as np
from torch import nn,optim
from model import RN
from functions import from_batch
from torch.autograd import Variable

# define hyperparameters
batch_size = 100
embed_size = 64
en_hidden_size = 32
mlp_hidden_size = 256
epochs = 100
qa = 20

# optional: if you're starting off from a previous model
startoff=80

# load dataset
with open('datasets/test_qa%d.txt' %qa) as f:
    lines=f.readlines()
if len(lines)%batch_size==0:
    num_batches = int(len(lines)/batch_size)
else:
    num_batches = int(len(lines)/batch_size)

# load vocabulary
word2idx = np.load('word2idx.npy').item()
idx2word = np.load('idx2word.npy').item()
vocab_size = len(word2idx)

if startoff>0:
    rn = torch.load('saved/rn_qa%d_epoch_%d_acc_0.970.pth' %(qa,startoff))
else:
    rn = RN(vocab_size, embed_size, en_hidden_size, mlp_hidden_size)
if torch.cuda.is_available():
    rn = rn.cuda()

total = 0
correct = 0

# training
for epoch in range(1):
    random.shuffle(lines) # shuffle lines
    for i in range(num_batches):
        batch = lines[i*batch_size:(i+1)*batch_size]
        S,Q,A = from_batch(batch)
        out = rn(S,Q)
        O = torch.max(out,1)[1].cpu().data.numpy().squeeze()
        score = np.array(O==A,int)
        total += len(score)
        correct += sum(score)
print("Total score: %d out of %d: %1.3f!!" % (correct,total,correct*100.0/total))
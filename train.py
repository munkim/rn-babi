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
epochs = 200
qa = 0

# optional: if you're starting off from a previous model
startoff=0

# load dataset
# with open('datasets/train_qa%d.txt' %qa) as f:
with open('datasets/train_10k.txt') as f:
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
    rn = torch.load('saved/rn_qa%d_epoch_%d.pth' %(qa,startoff))
else:
    rn = RN(vocab_size, embed_size, en_hidden_size, mlp_hidden_size)
if torch.cuda.is_available():
    rn = rn.cuda()
opt = optim.Adam(rn.parameters(),lr=2e-4)
criterion = nn.CrossEntropyLoss()

# for validation
def validate(rn, val_set):
    S, Q, A = from_batch(val_set)
    out = rn(S, Q)
    O = torch.max(out,1)[1].cpu().data.numpy().squeeze()
    score = np.array(O==A,int)
    total = len(score)
    correct = sum(score)
    return total,correct

# with open('datasets/test_qa%d.txt' %qa) as f:
with open('datasets/test_10k.txt') as f:
    val_set=f.readlines()[:batch_size]
    
# training
for epoch in range(epochs):
    random.shuffle(lines) # shuffle lines
    for i in range(num_batches):
        opt.zero_grad()
        batch = lines[i*batch_size:(i+1)*batch_size]
        S,Q,A = from_batch(batch)
        out = rn(S,Q)
        A = Variable(torch.LongTensor(A))
        if torch.cuda.is_available():
            A = A.cuda()
        loss = criterion(out,A)
        loss.backward()
        opt.step()
        if i % 20==0:
            print("loss for %d/%d: %1.3f" % (i,num_batches,loss.data[0]))
    print("loss for epoch %d: %1.3f" % (epoch,loss.data[0]))
#     if i % 5==0:
#         torch.save(obj=rn,f='saved/'+model_name)
    total, correct = validate(rn, val_set)
    val_score = correct*1.0/total
    print("Validation score for task %d: %1.3f"%(qa,val_score))
    model_name = 'rn_qa%d_epoch_%d_acc_%1.3f.pth' % (qa,epoch+startoff+1,val_score)
#    model_name = 'rn_10k_reduced_epoch_%d_acc_%1.3f.pth' % (qa,epoch+startoff+1,val_score)
    if epoch % 10==9:
        torch.save(obj=rn,f='saved/'+model_name)

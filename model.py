import torch
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import time
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, embed):
        super(Encoder, self).__init__()
        self.embed = embed

        self.lstm = nn.LSTM(input_size=embed_size,
            hidden_size=hidden_size, batch_first=True,
            bidirectional=False)

    def forward(self, input, lengths):
        # input(numpy): input tokens w/ padding, [total_sentences x max_seq_length]
        total_sentences = input.shape[0]
        max_length = input.shape[1]
        # lengths(list): lengths of individual lines, : [total_sentences]
        
        # 1. get lstm states of every line
        input = Variable(torch.LongTensor(input))
        if torch.cuda.is_available():
            input = input.cuda()
        embedded = self.embed(input) # [total_sentences, max_seq_length, embed_size] 
        states, _ = self.lstm(embedded) # out: [total_sentences x max_seq_length x hid]
        
        # 2. get masked region to indicate the length of every individual line
        mask = np.zeros([total_sentences,max_length])
        for i,j in enumerate(lengths):
            mask[i][j-1]=1
        mask = np.expand_dims(mask,axis=1) # [total_sentences, 1, max_length]
        mask = Variable(torch.Tensor(mask))
        if torch.cuda.is_available():
            mask = mask.cuda()
        states = torch.bmm(mask, states) #[total_sentences, 1, hidden]
        states = states.squeeze() # [total_sentences, hidden]
                
        return states

class MLP_G(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_G, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = F.relu(self.linear4(out))
        return out
    
class MLP_F(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size):
        super(MLP_F, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size*2)
        self.linear3 = nn.Linear(hidden_size*2, vocab_size)
    
    def forward(self, input):
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        return out

class RN(nn.Module):
    def __init__(self, vocab_size, embed_size, en_hidden_size, mlp_hidden_size):
        super(RN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encode_story = Encoder(vocab_size, embed_size, en_hidden_size, self.embed)
        self.encode_query = Encoder(vocab_size, embed_size, en_hidden_size, self.embed)
        self.mlp_g = MLP_G(en_hidden_size*3+2, mlp_hidden_size)
        self.mlp_f = MLP_F(mlp_hidden_size, mlp_hidden_size, vocab_size)
        self.start = time.time()
    
    def forward(self, story, query):
        s_input, s_lengths, s_sizes = story
        q_input, q_lengths = query
        
        # get [total_lines, hidden] encoded results of both stories and queries
        s_states = self.encode_story(s_input, s_lengths) # [total, hidden]
        q_states = self.encode_story(q_input, q_lengths) # [batch_size, hidden]
        
        # append relative position to s_states
        pos_info = []
        for s in s_sizes:
            pos_info.extend(np.ndarray.tolist(np.arange(s)+1)) # [total]
        pos_info = np.expand_dims(np.array(pos_info,dtype=float),1) # [total x 1]
        pos_info = Variable(torch.Tensor(pos_info))
        if torch.cuda.is_available():
            pos_info = pos_info.cuda()
        s_states = torch.cat([s_states,pos_info],1) # [total, hidden+1]

        # get object sets
        line_idx = 0
        obj_list = []
        for s in s_sizes:
            obj_list.append(s_states[line_idx:line_idx+s])
            line_idx += s
        # obj_list is a list where each item is [num_of_objects * (hidden+1)]
        out_list= []

        for b in range(len(q_states)): # b is for each item in a batch
            # for batch size, we now obtain each object value
            num_obj = len(obj_list[b])
            obj_set1 = obj_list[b].repeat(num_obj,1)
            obj_set2 = obj_list[b].repeat(1,num_obj).view(obj_set1.size())
            queries = q_states[b].repeat(num_obj*num_obj,1)
            # these three are all of size [num_objects^2, hidden(+1)]

            obj_set = torch.cat([obj_set1,obj_set2,queries],1)
            # size [num_objects^2, hidden*3+2]

            obj_set = self.mlp_g(obj_set).sum(0) # [hidden]
            out_list.append(obj_set)

        out = torch.cat(out_list,0) # [b x hidden]
        out = self.mlp_f(out) # [b x vocab_size]
        return out

    def update_time(self):
    	elapsed = time.time()
    	print("Time elapsed: ",elapsed-self.start)
    	self.start = elapsed
    	return

    def reset_time(self):
    	self.start = time.time()
    	return
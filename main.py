#setting logger
from logging import *
logger = getLogger("logger_50-59")
handler = StreamHandler()
handler.setLevel(DEBUG)
fmt = Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Sampler,Dataset
from torch.nn.utils.rnn import pad_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack

trainjp_path = "data/tok/kyoto-train.ja"
testjp_path = "data/tok/kyoto-test.ja"
trainen_path = "data/tok/kyoto-train.en"
testen_path = "data/tok/kyoto-test.en"

def make_dictionary(path1, path2):
  with open(path1) as f:
    string1 = f.readlines()
  with open(path2) as f:
    string1 += f.readlines()
  vocab = []
  for sentence in string1:
    vocab += sentence.strip().split()
  vocab = set(vocab)
  num_vocab = len(vocab)
  dic = dict()
  for i,w in enumerate(vocab):
    dic[w] = i+1
  dic["<bos>"] = num_vocab + 1
  dic["<eos>"] = num_vocab + 2
  return dic, num_vocab + 4
#special id
# padding : 0
# <bos>   : V + 1
# <eos>   : V + 2
# unknown : V + 3
# --------------
# total : V + 4

def string2id(dic,path):
  unknown = dic["<eos>"]
  with open(path) as f:
    string_x = f.readlines()
  id_x = []
  for x in string_x:
    x = ["<bos>"] + x.strip().split() + ["<eos>"]
    buf = []
    for w in x:
      try:
        buf.append(dic[w])
      except KeyError:
        buf.append(unknown)
    id_x.append(torch.tensor(buf))
  return id_x

def datasource(dic, path1, path2):
  xs = string2id(dic,path1)
  ys = string2id(dic,path2)
  ds = PackedDataset(xs,ys)
  return ds

class PackedDataset(Dataset):
  def __init__(self,x,y):
    assert len(x) == len(y), "Error: len x != len y"
    self.size = len(x)
    self.x = x
    self.y = y
    self.seq_x = [len(s) for s in x]
    self.seq_y = [len(s) for s in y]

  def __len__(self):
    return self.size

  def __getitem__(self,index):
    return{
        "x": self.x[index],
        "y": self.y[index],
        "seq_x" : self.seq_x[index],
        "seq_y" : self.seq_y[index]
        }
    #x : [tensor]
    #y : [tensor]
    #seq_x : [int]
    #seq_y : [int]
  
  def collate(self, samples):
    xs = [s["x"] for s in samples]
    ys = [s["y"] for s in samples]
    seq_x = [s["seq_x"] for s in samples]
    seq_y = [s["seq_y"] for s in samples]
    pad_x = pad(xs,batch_first=True)
    pad_y = pad(ys,batch_first=True)
    return{
        "x" : pad_x,
        "y" : pad_y,
        "seq_x" : seq_x,
        "seq_y" : seq_y
        }
    #x : padded_sequence
    #y : padded_sequence
    #seq_x : [int]
    #seq_y : [int]

class Encoder(nn.Module):
  def __init__(self,num_vocab,dim_emb,dim_hid):
    super(Encoder,self).__init__()
    self.dim_hid = dim_hid
    self.emb = nn.Embedding(num_vocab,dim_emb)
    self.lstm = nn.LSTM(input_size = dim_emb,
        hidden_size = dim_hid,
        batch_first = True,
        )
  def forward(self,batch):
    x = self.emb(batch["x"])
    x = pack(x,batch["seq_x"],batch_first=True,enforce_sorted=False)
    _, (h,_) = self.lstm(x)
    return h
  def initial_hidden(self,batch_size=1):
    return torch.zeros(1,1,self.dim_hid)

class Decoder(nn.Module):
  def __init__(self,dic,num_vocab,dim_emb,dim_hid):
    self.dim_hid = dim_hid
    super(Decoder,self).__init__()
    self.emb = nn.Embedding(num_vocab,dim_emb)
    self.lstm = nn.LSTM(input_size = dim_emb,
        hidden_size = dim_hid,
        batch_first = True,
        )
    self.lin = nn.Linear(dim_hid,num_vocab)

  # forward method is not for training
  def forward(self,h,id_bos,id_eos):
    y_0   = torch.tensor([[id_bos]])
    output = []
    count = 0
    id_predict = []
    c = self.initial_hidden()
    with torch.no_grad():
      y = self.emb(y_0)
      pos = 0
      while not pos == id_eos:
        if count > 100 : break
        out, (h,c) = self.lstm(y,(h,c))
        out = self.lin(out)
        y = torch.argmax(out,axis=2)
        id_predict.append(y.squeeze().item())
        count += 1
    return id_predict

  def train(self,batch,h):
    seq_y = batch["seq_y"]
    batch_y = batch["y"]
    y = self.emb(batch_y)
    c = self.initial_hidden(len(batch["y"]))
    pred, (_,_) = self.lstm(y,(h,c))
    y = self.lin(pred)
    pred = []
    target = []
    for i,len_seq in enumerate(seq_y):
      pred.append(y[i,:len_seq-1])
      target.append(batch_y[i,1:len_seq])
    return pred,target

  def initial_hidden(self,batch_size=1):
    return torch.zeros(1,batch_size,self.dim_hid)

NUM_BATCH = 100
NUM_EPOCH = 10
DIM_EMB = 100
DIM_HID = 50

def f90():
  train_path = (trainjp_path,trainen_path)
  test_path  = (testjp_path,testen_path)
  logger.debug("loading data")
  dic, num_vocab = make_dictionary(*train_path)
  train = datasource(dic,*train_path) 
  test = datasource(dic,*test_path) 
  logger.debug("loading data done")
  loader = DataLoader(train,
      collate_fn = train.collate,
      batch_size = NUM_BATCH,
      shuffle = True
      )
  encoder = Encoder(num_vocab, DIM_EMB,DIM_HID)
  decoder = Decoder(dic,num_vocab,DIM_EMB,DIM_HID)
  criterion = nn.CrossEntropyLoss()
  optimizer_encoder = optim.Adam(encoder.parameters())
  optimizer_decoder = optim.Adam(decoder.parameters())
  criterion = nn.CrossEntropyLoss()
  for i in range(NUM_EPOCH):
    loss_epoch = 0
    num_whole_sample = 0
    for batch in loader:
      loss = torch.tensor(0.,requires_grad=True)
      optimizer_encoder.zero_grad()
      optimizer_decoder.zero_grad()
      h = encoder(batch)
      pred,target = decoder.train(batch,h)
      for p,t in zip(pred,target):
        loss = loss + criterion(p,t)
      loss.backward()
      optimizer_encoder.step()
      optimizer_decoder.step()
      loss_epoch += (loss * len(batch))
      num_whole_sample += len(batch)
      logger.debug("batch done")
    logger.debug(loss_epoch / num_whole_sample)


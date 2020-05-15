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

batch_size = 4

def f90():
  train_path = (trainjp_path,trainen_path)
  test_path  = (testjp_path,testen_path)
  logger.debug("loading data")
  dic, num_vocab = make_dictionary(*train_path)
  train = datasource(dic,*train_path) 
  test = datasource(dic,*test_path) 
  logger.debug("loading data done")
  loader = DataLoader(train,collate_fn = train.collate,batch_size = batch_size,shuffle = True)
  logger.debug(iter(loader).next())

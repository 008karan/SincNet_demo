import os
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

import sys
import numpy as np
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool

np.set_printoptions(threshold=1e6)

pt_file="/home/gamut/Downloads/SincNet_test/sincnet_models/SincNet_TIMIT/model_raw.pkl"
# test
wav_lst_te="timit.wav"
#user_label={0:"adarsh",1:"anuj",2:"piyush",3:"peeyush",4:"sameer",5:"rajat",6:"rachna",7:"sangram",8:"shashikant",9:"karan",10:"eram",11:"anjani",12:"akash"}
#[windowing]
fs=int(16000)
cw_len=int(200)
cw_shift=int(10)

#[cnn]
cnn_N_filt=[80,60,60]
cnn_len_filt=[251,5,5]
cnn_max_pool_len=[3,3,3]
cnn_use_laynorm_inp=True
cnn_use_batchnorm_inp=False
cnn_use_laynorm=[True,True,True]
cnn_use_batchnorm=[False,False,False]
cnn_act=["leaky_relu","leaky_relu","leaky_relu"]
cnn_drop=[0.0,0.0,0.0]


#[dnn]
fc_lay=[2048,2048,2048]
fc_drop=[0.0,0.0,0.0]
fc_use_laynorm_inp=True
fc_use_batchnorm_inp=False
fc_use_batchnorm=[True,True,True]
fc_use_laynorm=[False,False,False]
fc_act=["leaky_relu","leaky_relu","leaky_relu"]

#[class]
class_lay= [462]
class_drop=[0.0]
class_use_laynorm_inp=False
class_use_batchnorm_inp=False
class_use_batchnorm=[False]
class_use_laynorm=[False]
class_act=["softmax"]


#[optimization]
lr=float(0.001)
batch_size=int(128)
N_epochs=int(80)
N_batches=int(800)
N_eval_epoch=int(8)
seed=int(1234)
 
# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

  
# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# Batch_dev
Batch_dev=128


# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.cuda()

DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }


DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()


if pt_file!='none':
   print('LOADING MODEL.')
   checkpoint_load = torch.load(pt_file)
   CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
   DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
   DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])
     
CNN_net.eval()
DNN1_net.eval()
DNN2_net.eval()
   
with torch.no_grad():  
    print(wav_lst_te)
    [signal, fs] = sf.read(wav_lst_te)
    signal=torch.from_numpy(signal).float().cuda().contiguous()
    
    beg_samp=0
    end_samp=wlen
     
    N_fr=int((signal.shape[0]-wlen)/(wshift))

    sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
    pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
    count_fr=0
    count_fr_tot=0
    while end_samp<signal.shape[0]:
      sig_arr[count_fr,:]=signal[beg_samp:end_samp]
      beg_samp=beg_samp+wshift
      end_samp=beg_samp+wlen
      count_fr=count_fr+1
      count_fr_tot=count_fr_tot+1
      if count_fr==Batch_dev:
        inp=Variable(sig_arr)
        pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
        count_fr=0
        sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
   
    if count_fr>0:
      inp=Variable(sig_arr[0:count_fr])
      pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))

    values, indices = torch.max(torch.sum(pout,0),0)
[val,best_class]=torch.max(torch.sum(pout,dim=0),0)
user_no=best_class.item()
avg_vec=pout.exp()
avg_vec=avg_vec.mean(dim=0)
print(avg_vec)
print(user_no)

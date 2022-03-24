import argparse
from padding import padding_data
import pickle
import numpy as np 
from evaluation import evaluation_model
from train import train_model
import time
import torch
import random
import os

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	#torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()



def read_args():
    parser = argparse.ArgumentParser()
     # Training our model
    parser.add_argument('-train', action='store_true', help='training DeepJIT model')  

    #parser.add_argument('-train_data', type=str, help='the directory of our training data')   
    #parser.add_argument('-dictionary_data', type=str, help='the directory of our dicitonary data')

    parser.add_argument('-do_valid', action='store_true', help='validing DeepJIT model')

    # Predicting our data
    parser.add_argument('-predict', action='store_true', help='predicting testing data')
    #parser.add_argument('-pred_data', type=str, help='the directory of our testing data')    

    # Predicting our data
    parser.add_argument('-load_model', type=str, help='loading our model')

    # Number of parameters for reformatting commits
    parser.add_argument('-msg_length', type=int, default=256, help='the length of the commit message')
    parser.add_argument('-code_line', type=int, default=10, help='the number of LOC in each hunk of commit code')
    parser.add_argument('-code_length', type=int, default=512, help='the length of each LOC of commit code')

    # Number of parameters for PatchNet model
    parser.add_argument('-embedding_dim', type=int, default=64, help='the dimension of embedding vector')
    parser.add_argument('-filter_sizes', type=str, default='1, 2, 3', help='the filter size of convolutional layers')
    parser.add_argument('-num_filters', type=int, default=64, help='the number of filters')
    parser.add_argument('-hidden_units', type=int, default=512, help='the number of nodes in hidden layers')
    parser.add_argument('-dropout_keep_prob', type=float, default=0.5, help='dropout for training DeepJIT')
    parser.add_argument('-l2_reg_lambda', type=float, default=5e-5, help='regularization rate')
    parser.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-num_epochs', type=int, default=30, help='the number of epochs')    
    parser.add_argument('-save-dir', type=str, default='model', help='where to save the snapshot')    

    parser.add_argument('-project', type=str, default='openstack')

    # CUDA
    parser.add_argument('-device', type=int, default=-1,
                        help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the GPU')
    return parser

if __name__ == '__main__':
    params = read_args().parse_args()
    

    params.dictionary_data = '../data/commit_cotents/processed_data/' + params.project + '/' +  params.project  +'_dict.pkl'
    params.train_data =  '../data/commit_cotents/processed_data/' + params.project + '/val_train/' + params.project + '_train.pkl'
    params.val_data = '../data/commit_cotents/processed_data/'  + params.project + '/val_train/' + params.project + '_val.pkl'  
    params.pred_data = '../data/commit_cotents/processed_data/' + params.project + '/' + params.project + '_test.pkl'
    
    params.save_dir = params.save_dir + '/' + params.project


    if params.train is True:
        data = pickle.load(open(params.train_data, 'rb'))
        ids, labels, msgs, codes = data 
        labels = np.array(labels)       
        #print('Train Data size:', len(labels))

        val_data = pickle.load(open(params.val_data, 'rb'))
        v_ids, v_labels, v_msgs, v_codes = val_data
        v_labels = np.array(v_labels)
        #print('Val Data size:', len(v_labels))

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')

        v_pad_msg = padding_data(data=v_msgs, dictionary=dict_msg, params=params, type='msg')
        v_pad_code = padding_data(data=v_codes, dictionary=dict_code, params=params, type='code')

        data = (pad_msg, pad_code, labels, dict_msg, dict_code)

        v_data = (v_pad_msg, v_pad_code, v_labels, dict_msg, dict_code)

        #train_model(data=data, val=v_data, params=params)        
        starttime = time.time()
        train_model(data=data, val=v_data, params=params)
        endtime = time.time()
        train_time = endtime - starttime
        print('training time needed:', train_time)
    
    elif params.predict is True:
        
        params.load_model = './model/' + params.project + '/best_model.pt'

        data = pickle.load(open(params.pred_data, 'rb'))
        ids, labels, msgs, codes = data 
        labels = np.array(labels)        

        dictionary = pickle.load(open(params.dictionary_data, 'rb'))   
        dict_msg, dict_code = dictionary

        pad_msg = padding_data(data=msgs, dictionary=dict_msg, params=params, type='msg')        
        pad_code = padding_data(data=codes, dictionary=dict_code, params=params, type='code')
        
        starttime = time.time()
        data = (pad_msg, pad_code, labels, dict_msg, dict_code)
        evaluation_model(data=data, params=params)
        endtime = time.time()
        pred_time = endtime - starttime 
        print('predicting time needed:', pred_time)
        

    else:
        print('--------------------------------------------------------------------------------')
        print('--------------------------Something wrongs with your command--------------------')
        print('--------------------------------------------------------------------------------')
        exit()



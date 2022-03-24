from model import DeepJIT
import torch 
from tqdm import tqdm
from utils import mini_batches_train, save, mini_batches_train_non_balance, mini_batches_test
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve
import torch.nn as nn
import os, datetime
import numpy as np

def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    return lr_auc


def train_model(data, val, params):
    data_pad_msg, data_pad_code, data_labels, dict_msg, dict_code = data
    v_pad_msg, v_pad_code, v_labels, dict_msg, dict_code = val


    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    params.save_dir = os.path.join(params.save_dir)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)    

    if len(data_labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = data_labels.shape[1]
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create and train the defect model
    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.l2_reg_lambda)
    criterion = nn.BCELoss()

    

    best_valid_score = 0
    early_stop_count = 5
    #print('Training Process:   completed epoches / the total epoches')
    for epoch in tqdm(range(1, params.num_epochs + 1)):
        total_loss = 0
        

        # building batches for training model
        batches = mini_batches_train(X_msg=data_pad_msg, X_code=data_pad_code, Y=data_labels)
        for i, (batch) in enumerate(batches):
            pad_msg, pad_code, labels = batch
            if torch.cuda.is_available():                
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(labels)
            else:            
                pad_msg, pad_code, labels = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()

            optimizer.zero_grad()
            predict = model.forward(pad_msg, pad_code)
            loss = criterion(predict, labels)
            total_loss += loss
            loss.backward()
            optimizer.step()

        
        
        if params.do_valid == True:
            # building batches for validation
            batches = mini_batches_test(X_msg=v_pad_msg, X_code=v_pad_code, Y=v_labels)
            model.eval()
            with torch.no_grad():
                all_predict, all_label = list(), list()
                for i, (batch) in enumerate(batches):
                    pad_msg, pad_code, label = batch
                    if torch.cuda.is_available():
                        pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                            pad_code).cuda(), torch.cuda.FloatTensor(label)
                    else:
                        pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                            labels).float()
                    if torch.cuda.is_available():
                        predict = model.forward(pad_msg, pad_code)
                        predict = predict.cpu().detach().numpy().tolist()
                    else:
                        predict = model.forward(pad_msg, pad_code)
                        predict = predict.detach().numpy().tolist()
                    all_predict += predict
                    all_label += labels.tolist()

            #print('validating data size:', len(all_predict))
            auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
            auc_pc_score = auc_pc(all_label, all_predict)
            print('Valid data -- AUC-ROC score:', auc_score,  ' -- AUC-PC score:', auc_pc_score)
            
            valid_score = auc_pc_score
            #print(' valid_score:',valid_score,'  best_valid_score:',  best_valid_score)
            if valid_score > best_valid_score:
                 best_valid_score = valid_score
                 print('save a better model', best_valid_score)
                 save(model, params.save_dir, 'epoch', epoch)
            else:
                print('no update of models', early_stop_count)
                if epoch > 5:
                    early_stop_count = early_stop_count - 1
                if early_stop_count < 0:
                    break
        


        

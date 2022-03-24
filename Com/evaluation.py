from model import DeepJIT
from utils import mini_batches_test
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, average_precision_score,  precision_recall_curve    
import torch 
from tqdm import tqdm
from matplotlib import pyplot
import numpy as np
import time


def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)
    #yhat = np.array(pred)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    #lr_f1 = f1_score(testy, yhat)
    #print(type(lr_precision), type(lr_recall))
    #print(np.shape(lr_precision), np.shape(lr_recall))
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    #print('AUC-PR:  auc=%.3f' % ( lr_auc))
    # plot the precision-recall curves

    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    return  lr_auc


def evaluation_model(data, params):
    pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_test(X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJIT(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))


    starttime  = time.time()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
    
    endtime = time.time()
    dtime = endtime -starttime
    #print('all test data size:', len(all_predict))
      
    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    pc = auc_pc(all_label, all_predict)
    
    import pandas as pd
    df = pd.DataFrame({'label': all_label, 'pred': all_predict})
    df.to_csv('./pred_scores/test_com_' + params.project + '.csv', index=False, sep=',')
    
    #print('Time cost:', dtime)

    #print('Test data -- AUC-ROC score:', auc_score,  ' -- AUC-PC score:', pc)

    
    threshold = [0.5] 
    for t in threshold:
        real_pred = [1 if p > t else 0 for p in all_predict]
        f1 = f1_score(y_true=all_label, y_pred=real_pred)
    
    print("AUC-ROC:{}  AUC-PR:{}  F1-Score:{}".format(auc_score, pc, f1))
    

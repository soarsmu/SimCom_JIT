from itertools import groupby
import csv
from collections import Counter
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from matplotlib import pyplot
import argparse
import time

np.random.seed(10)



def read_csv_1 (fname):

    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(int(line[0]))
            la.append(int(line[1]))
            pred.append(float(line[2]))
            
        
        
    #print(len(pred), len(label), len(la))
    return pred, label, la


def read_csv_2 (fname):

    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(line[0])
            pred.append(float(line[1]))
            
              
    #print(len(pred), len(label))
    return pred, label



def eval_(y_true,y_pred, thresh=None):
    

    #print('size:', len(y_true), len(y_pred))
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    #auc_pc(y_true, y_pred)
    if thresh != None:
        y_pred = [ 1.0 if p> thresh else 0.0 for p in y_pred]
        
    
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print('AUC:', auc)
    
    



## AUC-PC
# predict class values
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

    return lr_auc


parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='openstack')
args = parser.parse_args()


data_dir1 = "./Com/pred_scores/"
data_dir2 = "./Sim/pred_scores/"

project = args.project

#Com
com_ = data_dir1+ 'test_com_' + project +'.csv'

#Sim
sim_ = data_dir2 + 'test_sim_' + project +'.csv'


## LAPredict
pred, label = read_csv_2 (sim_)
'''
label = [float(l) for l in label]
#print('\n Sim:')
mean_pred = float(sum(pred)/len(pred))
#eval_(y_true=np.array(label),  y_pred=np.array(pred), thresh=mean_pred)
auc_pc(label, pred)
avg_pr = average_precision_score(label, pred)
#print('average_precision_score:', avg_pr)
t = 0.5
#print('threshold:',t)
real_pred = [1 if p > t else 0 for p in pred]
y_true = label
f1_ = f1_score(y_true=y_true,  y_pred=real_pred)
acc = accuracy_score(y_true=y_true, y_pred=real_pred)
prc = precision_score(y_true=y_true, y_pred=real_pred)
rc = recall_score(y_true=y_true, y_pred=real_pred)
#print("Threshold: {}   F1-Score:{}  ".format(t, f1_))
'''



##DeepJIT 
pred_, label_ = read_csv_2 (com_)
'''
label_ = [float(l) for l in label_]


#print('\n Com: ')
mean_pred = float(sum(pred)/len(pred))
#eval_(y_true=np.array(label_),  y_pred=np.array(pred_), thresh=mean_pred)
auc_pc(label_, pred_)
avg_pr = average_precision_score(label_, pred_)
#print('average_precision_score:', avg_pr)   



t = 0.5
#print('threshold:',t)
real_pred = [1 if p > t else 0 for p in pred_]


y_true = label_
f1_ = f1_score(y_true=y_true,  y_pred=real_pred)
acc = accuracy_score(y_true=y_true, y_pred=real_pred)
prc = precision_score(y_true=y_true, y_pred=real_pred)
rc = recall_score(y_true=y_true, y_pred=real_pred)
#print("Threshold: {}   F1-Score:{} ".format(t, f1_))
'''



## Simple add
pred2 = [ pred_[i] + pred[i] for i in range(len(pred_))]
#print(len(pred2), len(label_))
auc2 = roc_auc_score(y_true=np.array(label_),  y_score=np.array(pred2))
#print('\n SimCom: ')
mean_pred = float(sum(pred2)/len(pred2))
#eval_(y_true=np.array(label_),  y_pred=np.array(pred2), thresh = mean_pred )
pc_ = auc_pc(label_, pred2)

t = 1
real_label = [float(l) for l in label_]
real_pred = [1 if p > t else 0 for p in pred2]
f1_ = f1_score(y_true=real_label,  y_pred=real_pred)
print("AUC-ROC:{}  AUC-PR:{}  F1-Score:{}".format(auc2, pc_, f1_))

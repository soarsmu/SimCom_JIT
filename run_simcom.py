import numpy 
import os
import re
import argparse



def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-project', type=str, default='openstack')
    return parser

params = read_args().parse_args()
project = params.project



run_sim = "python sim_model.py  -project {} "
train_com = "python main.py  -train -project {} -do_valid"
test_com = "python main.py  -predict -project {}"
run_simcom = "python combination.py -project {}"




## Train & Predict for Sim
current_wrorking_dir = os.popen('pwd').read()
os.chdir(current_wrorking_dir.strip()  + "/Sim")
cmd = run_sim
result = os.popen(cmd.format(project)).readlines()
#print(result)
print('Training of Sim is finished')



## Train Com
os.chdir(current_wrorking_dir.strip()  + "/Com") 
cmd = train_com
result = os.popen(cmd.format(project)).readlines()
print('Training of Com is finished')  

## Predict by Com
cmd = test_com
result = os.popen(cmd.format(project)).readlines()
#print(result)


## Model fusion of Sim and Com
print('The final results for SimCom: \n')
os.chdir(current_wrorking_dir.strip())
cmd = run_simcom
result = os.popen(cmd.format(project)).readlines()
print(result)









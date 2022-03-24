
import pickle 
from collections import Counter


data_path = '/data/ISSTA21-JIT-DP/CC2Vec/data/'
project = 'openstack' 
split = 'test' 
#file_name = data_path + project + '/50k/cc2vec/'+  project + '_' + split+ '.pkl'
file_name = data_path + project + '/deepjit/'+  project + '_' + split+ '.pkl'
data = pickle.load(open(file_name, 'rb'))
print(len(data))
ids, labels, msgs, codes = data
print(len(ids))

print(ids[0])
print(labels[0])
print(codes[3])
print(len(codes[3]))
print(msgs[0])



### --------- Statistics of the Datasets ------------- 

## Code Changes

# How many hunks per commit
len_code = []
for code in codes:
    len_code.append(len(code))

print(len(len_code))
result = Counter(len_code)
print(result)
n = dict()
for k,v in result.items():
    v_ = str(int(v/len(len_code)*100)) + '%'
    n[k] = v_

print('How many hunks per commit: ')
cu = 0
for key in sorted(n):
    cu += int(n[key].split('%')[0])
    print(key, n[key], str(cu) + '%')


# How many words per hunk
print('\n')
len_tokens = []
tokens = []
for code in codes:
    #print(code)
    for d in code:
        #print(d)
        add = ' <n> '.join(d['added_code'])
        delete = ' <n> '.join(d['removed_code'])
        #print(d['added_code'] + d['removed_code'])
        add_remove = add+delete
        #print(len(add_remove.split()))
        #print(len(add_remove))
        #print(add_remove)
        #print(add_remove.split())
        len_tokens.append(len(add_remove.split()))
        tokens.extend(add_remove.split())
        #print(len_tokens)
        #print(len(tokens))

print(len(len_tokens), len_tokens[0:1000])

from itertools import groupby
print('\n How many words per hunk')
cu = 0
for k, g in groupby(sorted(len_tokens), key=lambda x: x//100):
        #cu += len(list(g))/len(len_tokens)
        print('{}-{}: {}'.format(k*100, (k+1)*100-1, float(len(list(g))/len(len_tokens)) ))

# Dict size of Code
print(len(tokens))
print('code dict size:', len(list(set(tokens))))


## Commit Msg

# How many msg words per commit
msg_len = list()
msg_tokens = list()
for msg in msgs:
    #print(msg)
    msg_len.append(len(msg.split()))
    msg_tokens.extend(msg.split())
    #print(msg.split())

print('\n How many msg words per commit')
for k, g in groupby(sorted(msg_len), key=lambda x: x//50):
    print('{}-{}: {}'.format(k*50, (k+1)*50-1, float(len(list(g))/len(msg_len)) ))
# Dict Size of Msg
print(len(msg_tokens))
print('msg dict size:', len(list(set(msg_tokens))))




## Label
print('\n Label distribution')
result = Counter(labels)
print(result)



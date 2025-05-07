"""
Takes csv file and returns tokenized files
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import CamembertTokenizer
import pickle 

path = "/pasteur/appa/homes/gbizelbi/comcor/Gaston/"
data_file = path + 'data/data.csv'
model_file = path + 'camembert/camembert-base/'
output_data_path = path + 'data/vectorized/'

test_size=0.2
val_size=0.1
stratify=None
MAX_SEQ_LEN = 100


print("Load data......", flush=True)

# load data
df = pd.read_csv(data_file, index_col = False)

# filter only SI responses
df = df[df.txt_situation == 'SI'][['txt_raw', 'evenement_suspect_contexte']]

# renames columns
df.rename({'txt_raw': 'txt',
           'evenement_suspect_contexte': 'label'},
          axis=1, inplace=True)

# transform labels to numerical values
label_to_num = {'Professionnel':0,
                'Familial':1,
                'Amical':2,
                'Sportif':3,
                'Culturel':4,
                'Religieux':5,
                'Dans un autre contexte':6}

df['label'] = df['label'].astype('category').cat.rename_categories(label_to_num)

print("n responses: ",len(df), flush=True)
print('Splitting.....', flush = True)

train, test = train_test_split(df, test_size=test_size,random_state=42,shuffle=True, stratify=stratify)
df_train, df_val = train_test_split(train, test_size=val_size,random_state=42,shuffle=True, stratify=stratify)

train_responses = df_train['txt'].astype(str)
val_responses = df_val['txt'].astype(str)
test_responses = test['txt'].astype(str)

train_labels = df_train['label'].astype(int)
val_labels = df_val['label'].astype(int)
test_labels = test['label'].astype(int)

print("size train, val, test:",len(train_labels), len(val_labels), len(test_labels), flush=True)
print("Tokenization...........", flush=True)

#CamembertTokenizer
tokenizer = CamembertTokenizer.from_pretrained(model_file, local_files_only=True)


responses_len = [len(tokenizer.encode(response, max_length=512))
                          for response in train_responses]

short_responses = sum(np.array(responses_len) <= MAX_SEQ_LEN)
long_responses = sum(np.array(responses_len) > MAX_SEQ_LEN)

print("{} responses with LEN > {} ({:.2f} % of total data)".format(
    long_responses,
    MAX_SEQ_LEN,
    100 * long_responses / len(responses_len)
    ), flush=True)


def encode_responses(tokenizer, responses, max_length):
    token_ids = np.zeros(shape=(len(responses), max_length),
                         dtype=np.int32)
    for i, response in enumerate(responses):
        encoded = tokenizer.encode(response, max_length=max_length)
        token_ids[i, 0:len(encoded)] = encoded
    attention_mask = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_mask": attention_mask}


encoded_train = encode_responses(tokenizer, train_responses, MAX_SEQ_LEN)
encoded_valid = encode_responses(tokenizer, val_responses, MAX_SEQ_LEN)
encoded_test = encode_responses(tokenizer, test_responses, MAX_SEQ_LEN)

# Labels
y_train = np.array(train_labels)
y_val = np.array(val_labels)
y_test = np.array(test_labels)

def pickle_file(file, file_name):
    with open(output_data_path+file_name+'.pkl', 'wb') as f:
        pickle.dump(file, f)
    return None

pickle_file(encoded_train, 'encoded_train')
pickle_file(encoded_valid, 'encoded_valid')
pickle_file(encoded_test, 'encoded_test')

pickle_file(y_train, 'y_train')
pickle_file(y_val, 'y_val')
pickle_file(y_test, 'y_test')
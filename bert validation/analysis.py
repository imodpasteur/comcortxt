import numpy as np
import pickle
import tensorflow as tf
from transformers import TFCamembertForSequenceClassification

from time import time
import datetime


def runtime(t_init, t_final):
    str(datetime.timedelta(seconds=t_init-t_final))[:-7]

t_init = time()

path = "/pasteur/appa/homes/gbizelbi/comcor/Gaston/"
data_path = path + 'data/vectorized/'
weights_path = path + 'code/output/weights'
model_file = path + 'camembert/tf-camembert-base/'

print('Loading Data...', flush=True)

def load_pickle(file_name):
    with open(data_path + file_name+'.pkl', 'rb') as f:
        file = pickle.load(f)
    return file

encoded_train = load_pickle('encoded_train')
encoded_valid = load_pickle('encoded_valid')
encoded_test = load_pickle('encoded_test')

y_train = load_pickle('y_train')
y_val = load_pickle('y_val')
y_test = load_pickle('y_test')

print('Import model...', flush=True)

model = TFCamembertForSequenceClassification.from_pretrained(model_file, local_files_only=True)

opt = tf.keras.optimizers.SGD(learning_rate=5e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)    
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

latest_weights = tf.train.latest_checkpoint(weights_path)

print('latest_weights:', latest_weights, flush=True)

model.load_weights(latest_weights)

print('Scoring...', flush=True)

scores = model.predict(encoded_valid)

scores_test = model.predict(encoded_test)

def pickle_file(file, file_name):
    with open(data_path+file_name+'.pkl', 'wb') as f:
        pickle.dump(file, f)
    return None

pickle_file(scores['logits'], 'y_pred')
pickle_file(scores_test['logits'], 'y_pred_test')


t_final = time()

runtime(t_init, t_final)
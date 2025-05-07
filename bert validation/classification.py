from transformers import TFCamembertForSequenceClassification
import tensorflow as tf
import pickle

path = "/pasteur/appa/homes/gbizelbi/comcor/Gaston/"
data_path = path + 'data/vectorized/'
model_file = path + 'camembert/tf-camembert-base/'

assert tf.__version__ >= "2.0"
print("tensorflow version: ", tf.__version__)

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

print("Data Loaded. Size of train, val, test:",len(y_train), len(y_val), len(y_test), flush=True)

print('Loading Model...', flush=True)

model = TFCamembertForSequenceClassification.from_pretrained(model_file, local_files_only=True)

opt = tf.keras.optimizers.SGD(learning_rate=5e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)    

model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

initial_weights = model.get_weights()
model.summary()

print('Training Model...', flush=True)

history = model.fit(
    encoded_train, y_train, epochs=10, batch_size=256, 
    validation_data=(encoded_valid, y_val), verbose=1
)

model.save_weights('output/weights/camembert_comcor_weights')
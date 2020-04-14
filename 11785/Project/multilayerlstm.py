#%%
from __future__ import print_function
import numpy as np
import math
import scipy.io as sio
from keras.models import Model, load_model, Sequential
from keras.layers import Input, LSTM, Dense, LSTMCell, RNN, Conv1D, Concatenate, TimeDistributed
from keras import optimizers
from keras import losses
from keras import regularizers
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


ktf.set_session(get_session())


#%%
## Before Training

caseNo = 118 # number of buses (nodes) in the grid network
input_dim = caseNo * 2 # dim(y)
weight_4_mag = 100.0 # cuz magnitutes are normalized
weight_4_ang = 1.0 

# data loading
data_path = '/home/yihe/Data/IEEE_118_bus_FASE_dataset.mat'
psse_data = sio.loadmat(data_path) # read data
data_x = psse_data['inputs'] #inputs = noisy measurements (z in the write-up)
data_y = psse_data['labels'] #outputs = system states (x in the write-up)

# scale the magnitudes
data_y[0:caseNo,:] = weight_4_mag*data_y[0:caseNo,:]
data_y[caseNo:,:] = weight_4_ang*data_y[caseNo:,:]

#%%
# seperate them into training 80%, test 20%
split_train = int(0.8*psse_data['inputs'].shape[1])
split_val = psse_data['inputs'].shape[1] - split_train
train_x = np.transpose(data_x[:, :split_train])
train_y = np.transpose(data_y[:, :split_train])
val_x   = np.transpose(data_x[:, split_train:split_train+split_val])
val_y   = np.transpose(data_y[:, split_train:split_train+split_val]) # dim = (,236)
test_x  = np.transpose(data_x[:, split_train+split_val:])
test_y  = np.transpose(data_y[:, split_train+split_val:])

total_v = len(train_y)
print(train_y.shape, val_y.shape, val_y[0].shape)

#How many timesteps e.g sequence length
time_steps = 5
counter = total_v - 2*time_steps

#Input data
vX = []
#output data
y = []

for i in range(counter):
    #This one goes from 0-100 so it gets 100 values starting from 0 and stops
    #just before the 100th value
    theInput = train_y[i:i+time_steps]
    theOutput = train_y[i+time_steps:i+2*time_steps]
    vX.append(theInput)
    y.append(theOutput)

# Inputs as sequences of length time_steps
X = np.reshape(vX, (len(vX), time_steps, input_dim)) # dim = (14816,5,236)
print(X.shape)
# Outputs as sequence of length time_steps
y = np.reshape(y, (len(y), time_steps, input_dim)) # dim = (14816,5,236)
print(y.shape)
# Now you can define the model e.g. Model = LSTM(X.shape, y.shape[1], weights=None)
#%%
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
#%%

#Hyperparameters
batch_size = 8
epochs = 30
learning_rate = 0.01
decay = 5e-5
momentum = 0.9
convs = [128]
padding = 'same'
kernel_size = 1
stride = 1
layers = [256, 256, 256]
linears = [512]
regulariser = None
input_dim = X.shape[-1]
output_dim = y.shape[-1]
optimizer = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum)
loss = 'mse'
mode = 'Train'

encoder_inputs = Input(shape=(None, input_dim))

conv_extractor = Sequential()
for filter_dim in convs:
    conv_extractor.add(Conv1D(filter_dim, kernel_size = kernel_size, strides = stride, padding='valid'))

encoder_features = conv_extractor(encoder_inputs)

encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(LSTMCell(hidden_neurons, kernel_regularizer=regulariser, recurrent_regularizer=regulariser, bias_regularizer=regulariser))

encoder = RNN(encoder_cells, return_sequences=True, return_state=True)
encoder_outputs_and_states = encoder(encoder_features)
encoder_outs = encoder_outputs_and_states[0]
encoder_states = encoder_outputs_and_states[1:]

decoder_inputs = Input(shape=(None, input_dim))
decoder_features = conv_extractor(decoder_inputs)
decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(LSTMCell(hidden_neurons, kernel_regularizer=regulariser, recurrent_regularizer=regulariser, bias_regularizer=regulariser))

decoder = RNN(decoder_cells, return_sequences=True, return_state=True)
decoder_outputs_and_states = decoder(decoder_features, initial_state=encoder_states)
decoder_outputs = decoder_outputs_and_states[0]

decoder_dense = Sequential()
for hidden_sizes in linears:
    decoder_dense.add(Dense(hidden_sizes, activation='relu'))
decoder_dense.add(Dense(output_dim, activation='linear'))
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimizer, loss=loss)

#%%
encoder_input_data = X
decoder_input_data = np.zeros_like(encoder_input_data)
decoder_input_data[:,1:,:] = X[:,:4,:]
decoder_target_data = y

if mode == 'Train':
    sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum)
    model.compile(optimizer=sgd, loss='mean_absolute_error')
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs, 
                        validation_split=0.2)
    model.save('lstmed.h5')
else:
    model = load_model('lstmed.h5')

#%%
encoder_inputs = model.inputs[0]
encoder_outputs = model.layers[3].output
encoder_states = encoder_outputs[1:]
encoder_predict_model = Model(encoder_inputs, encoder_states)

decoder_states_inputs = []

for hidden_neurons in layers[::-1]:
    decoder_states_inputs.append(Input(shape=(hidden_neurons,)))
    decoder_states_inputs.append(Input(shape=(hidden_neurons,)))

decoder_inputs = model.inputs[1]
conv_extractor = model.layers[2]
decoder_features = conv_extractor(decoder_inputs)
decoder = model.layers[4]
decoder_outputs_and_states = decoder(decoder_features, initial_state=decoder_states_inputs)

decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states[1:]

decoder_dense = model.layers[5]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_predict_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

#%%
## After Training
# Also, below is a function that compute the normalized RMSE, feel free to use it
ans = [] # ans collects the result of all the voltages
w = np.zeros((2,236))
test_no = 3706 - time_steps #Note that 3706 is the number of examples of the test/(validation) set
v_start = vX[-1] # dim = (5,236)

for i in range(int(test_no/time_steps+1)):

    i *= time_steps
    input_seq = np.reshape(v_start, (1, time_steps, input_dim)) # dim = (1,5,236)

    # # Encode the input as state vectors.
    states_value = encoder_predict_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, output_dim))
    
    # Populate the first target sequence with end of encoding series pageviews
    target_seq[0, 0, :] = input_seq[0, -1, :]

    # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
    # (to simplify, here we assume a batch of size 1).

    decoded_seq = np.zeros((1,time_steps,input_dim))
    pred = np.zeros((1, time_steps, output_dim))
    for i in range(time_steps):
        
        outputs = decoder_predict_model.predict([target_seq] + states_value)
        output = outputs[0]
        pred[0, i, :] = output
        decoded_seq[0,i,:] = output[0,0,:]

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, output_dim))
        target_seq[0, 0, :] = output[0,0,:]

        # Update states
        states_value = outputs[1:]
        
        # print(output.shape)

    # print('Done!')
    # pred = model.predict([input_seq, decoder_input]) # dim = (1,5,236)
    # print(np.shape(pred))
    w = np.concatenate((w, np.reshape(pred, (time_steps, input_dim))), axis = 0) # dim = (5,236)
    # print(np.shape(w))
    true_v = np.reshape(val_y[i:i+time_steps], (time_steps, input_dim)) # dim = (5,236)
    # print(np.shape(true_v))
    v_start = np.concatenate((v_start, true_v), axis = 0) # dim = (10,236)
    # print(np.shape(v_start))
    v_start = v_start[time_steps:] # dim = (5,236)
    # print(np.shape(v_start))

#%%
voltage_distance = np.zeros((test_no,caseNo)) # dim = (3706,118)
voltage_norm = np.zeros((test_no,1)) # dim = (3706,1)
val_predic = np.reshape(w[:test_no,:], (test_no, input_dim)) # dim = (3706,236)


for i in range(test_no):
    for j in range(caseNo):
        predic_r, predic_i = (1/weight_4_mag)* val_predic[i, j]*math.cos(val_predic[i, j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_predic[i,j]*math.sin(val_predic[i, j+caseNo]*2*math.pi/360)
        val_r, val_i = (1/weight_4_mag)*val_y[i,j]*math.cos(val_y[i,j+caseNo]*2*math.pi/360), (1/weight_4_mag)*val_y[i][j]*math.sin(val_y[i][j+caseNo]*2*math.pi/360)
        voltage_distance[i,j] = (predic_r-val_r)**2 + (predic_i-val_i)**2

    voltage_norm[i,] = (1/caseNo)*np.sqrt(np.sum(voltage_distance[i,:]))
print("\n NRMSE = : %.4f%%" % (np.mean(voltage_norm)*100))

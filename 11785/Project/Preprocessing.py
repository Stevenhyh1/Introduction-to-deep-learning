#%%
from __future__ import print_function
import numpy as np
import math
import scipy.io as sio
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras import optimizers
from keras import losses


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

#Hyperparameters
batch_size = 64
epochs = 20
learning_rate = 0.01
decay = 5e-5
momentum = 0.9
latent_dim = 256
input_dim = X.shape[-1]
output_dim = y.shape[-1]
mode = 'Train'

# Define an input series and encode it with an LSTM. 
encoder_inputs = Input(shape=(None, input_dim)) 
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the final states. These represent the "context"
# vector that we use as the basis for decoding.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
# This is where teacher forcing inputs are fed in.
decoder_inputs = Input(shape=(None, input_dim)) 

# We set up our decoder using `encoder_states` as initial state.  
# We return full output sequences and return internal states as well. 
# We don't use the return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(output_dim) # 1 continuous output at each timestep
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

#%%
encoder_input_data = X
decoder_input_data = np.zeros_like(encoder_input_data)
decoder_target_data = y

if mode == 'Train':
    sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum)
    model.compile(optimizer=sgd, loss='mean_absolute_error')
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)
    model.save('s2s2.h5')
else:
    model = load_model('s2s.h5')
#%%

# # from our previous model - mapping encoder sequence to state vectors
# encoder_model = Model(encoder_inputs, encoder_states)

# # A modified version of the decoding stage that takes in predicted target inputs
# # and encoded state vectors, returning predicted target outputs and decoder state vectors.
# # We need to hang onto these state vectors to run the next step of the inference loop.
# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]

# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model([decoder_inputs] + decoder_states_inputs,
#                       [decoder_outputs] + decoder_states)

#%%
## After Training
# Also, below is a function that compute the normalized RMSE, feel free to use it
ans = [] # ans collects the result of all the voltages
w = np.zeros((2,236))
test_no = 3706 - time_steps #Note that 3706 is the number of examples of the test/(validation) set
v_start = vX[-1] # dim = (5,236)
decoder_input = np.zeros((1,5,236))

for i in range(int(test_no/5+1)):

    i *= 5
    input_seq = np.reshape(v_start, (1, time_steps, input_dim)) # dim = (1,5,236)

    # # Encode the input as state vectors.
    # states_value = encoder_model.predict(input_seq)

    # # Generate empty target sequence of length 1.
    # target_seq = np.zeros((1, 1, output_dim))
    
    # # Populate the first target sequence with end of encoding series pageviews
    # target_seq[0, 0, :] = input_seq[0, -1, :]

    # # Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
    # # (to simplify, here we assume a batch of size 1).

    # decoded_seq = np.zeros((1,time_steps,input_dim))
    # pred = np.zeros((1, time_steps, output_dim))
    # for i in range(time_steps):
        
    #     output, h, c = decoder_model.predict([target_seq] + states_value)
    #     pred[0, i, :] = output
    #     decoded_seq[0,i,:] = output[0,0,:]

    #     # Update the target sequence (of length 1).
    #     target_seq = np.zeros((1, 1, output_dim))
    #     target_seq[0, 0, :] = output[0,0,:]

    #     # Update states
    #     states_value = [h, c]
        
        # print(output.shape)
    # print('Done!')
    pred = model.predict([input_seq, decoder_input]) # dim = (1,5,236)
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

# %%

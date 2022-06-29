import numpy as np
import math

# create RNN architecture
learning_rate = 0.0001
seq_len = 50
max_epochs = 25
hidden_dim = 100
output_dim = 1
bptt_truncate = 5 # backprop through time --> lasts 5 interations
min_clip_val = -10
max_clip_val = 10

def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
    X -> [Matrix] Data 
    Y -> [Matrix] Output 
    U -> [Matrix] Weights from the input layer to the hidden layer
    V -> [Matrix] Weights from the hidden layer to the output Layer
    W -> [Matrix] Recurrent weights from the hidden layer to itself
"""
def calculate_loss(X, Y, U, V, W):
    
    loss = 0.0

    for i in range(Y.shape[0]):

        x, y = X[i], Y[i]
        prev_activation = np.zeros((hidden_dim, 1)) # value of previous
                                                    # activation
        
        for timestep in range(seq_len):

            new_input = np.zeros(x.shape) # forward pass, done for each 
                                          # step in the sequence

            new_input[timestep] = x[timestep] # define a single 
                                              # input for that timestep

            mul_u = np.dot(U, new_input)
            mul_w = np.dot(W, prev_activation)

            _sum = mul_u + mul_w

            activation = sigmoid(_sum)

            mul_v = np.dot(V, activation)

            prev_activation = activation

        loss_per_record = float((y - mul_v) ** 2/2)
        loss += loss_per_record
    
    return loss, activation


"""
    x -> [Vector] Input for this data point 
    U -> [Matrix] Weights from the input layer to the hidden layer
    V -> [Matrix] Weights from the hidden layer to the output Layer
    W -> [Matrix] Recurrent weights from the hidden layer to itself
    prev_activation -> [Matrix] Previous activation for the final layer
"""
def calc_timestep_activation(x, U, V, W, prev_activation):
    timesteps_activations = []

    for timestep in range(seq_len):
        new_input = np.zeros(x.shape)
        new_input[timestep] = x[timestep]
        mul_u = np.dot(U, new_input)
        mul_w = np.dot(W, prev_activation)
        _sum = mul_w + mul_u
        activation = sigmoid(_sum)
        mul_v = np.dot(V, activation)
        timesteps_activations.append({'activation':activation,
                                    'prev_activation': prev_activation})
        prev_activation = activation
    
    return timesteps_activations, mul_u, mul_w, mul_v


"""
    x -> [Vector] Input for this data point
    U -> [Matrix] Weights from the input layer to the hidden layer
    V -> [Matrix] Weights from the hidden layer to the output Layer
    W -> [Matrix] Recurrent weights from the hidden layer to itself
    d_mul_v -> The differential for the last layer
    mul_u -> The input value to the hidden layer
    mul_w -> The input value to the hidden layer
    timestep_actv -> The list of activations through timesteps
"""
def backprop(x, U, V, W, d_mul_v, mul_u, mul_w, timestep_actv):
    
    # First, set up the differential for each layer
    dU = np.zeros(U.shape)
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)

    # Then, set up the differential for each layer in the timestep
    dU_t = np.zeros(U.shape)
    dV_t = np.zeros(V.shape)
    dW_t = np.zeros(W.shape) 

    # Next, we'll set up the differential for the truncated 
    # backpropagation through time
    dU_i = np.zeros(U.shape)
    dW_i = np.zeros(W.shape) 

    # Finally, we'll set up the input weights of the hidden layer as the most
    # recent sum of the U and V matrix weight outputs and the differential of
    # the last layer
    _sum = mul_u + mul_w
    d_s_v = np.dot(np.transpose(V), d_mul_v)

    # We'll define to calculate the differential for the previous activation
    # of the hidden layer.
    def get_previous_activation_differential(_sum, ds, W):
        d_sum = _sum * (1 - _sum) * ds
        d_mul_w = d_sum * np.ones_like(ds)
        return np.dot(np.transpose(W), d_mul_w)

    # Next, we'll create the differential for this recurrent timestep by 
    # getting the dot product of the hidden weights and the timestep's previous
    # activation. After this, we'll do the same step we do in the forward pass
    # by creating a new input for this recurrent timestep. This give us the
    # differential for the input layer for this recurrent timestep. Finnaly, 
    # we'll increment the differential values for the hidden layer and the 
    # input layer with the differentials for the recurrent timestep.

    for timestep in range(seq_len):

        dV_t = np.dot(d_mul_v, 
                    np.transpose(timestep_actv[timestep]['activation']))
        ds = d_s_v
        d_prev_activ = get_previous_activation_differential(_sum, ds, W)

        for _ in range(timestep-1, max(-1, timestep-bptt_truncate-1), -1):
            ds = d_s_v + d_prev_activ
            d_prev_activ = get_previous_activation_differential(_sum, ds, W)
            dW_i = np.dot(W, 
                    timestep_actv[timestep]['prev_activation'])
            
            new_input = np.zeros(x.shape)
            new_input[timestep] = x[timestep]
            dU_i = np.dot(U, new_input)

            dU_t += dU_i
            dW_t += dW_i
        
        dU += dU_t
        dV += dV_t
        dW += dW_t

        # take care of possible exploding gradients

        if dU.max() > max_clip_val:
            dU[dU > max_clip_val] = max_clip_val
        if dV.max() > max_clip_val:
            dV[dV > max_clip_val] = max_clip_val
        if dW.max() > max_clip_val:
            dW[dW > max_clip_val] = max_clip_val

        if dU.min() < min_clip_val:
            dU[dU < min_clip_val] = min_clip_val
        if dV.min() < min_clip_val:
            dV[dV < min_clip_val] = min_clip_val
        if dW.min() < min_clip_val:
            dW[dW < min_clip_val] = min_clip_val

    return dU, dV, dW


# training model
def train(U, V, W, X, Y, X_validation, Y_validation):

    for epoch in range(max_epochs):
        # calculate initial loss, ie  what the output is given a random set of 
        # weights

        loss, prev_activation = calculate_loss(X, Y, U, V, W)

        # check validation loss 
        val_loss, _ = calculate_loss(X_validation, Y_validation, U, V, W)

        print(f'Epoch: {epoch+1}, Loss: {loss}, Validation Loss: {val_loss}')

        # train model/forward pass
        for i in range(Y.shape[0]):

            x, y = X[i], Y[i]
            # Activation for each timestep
            timestep_actv, mul_u, mul_w, mul_v = calc_timestep_activation(
                x, U, V, W, prev_activation
            )

            # difference of the prediction
            d_mul_v = mul_v - y
            dU, dV, dW = backprop(x, U, V, W, d_mul_v, mul_u, mul_w, timestep_actv)

            # update weights
            U -= learning_rate * dU
            V -= learning_rate * dV
            W -= learning_rate * dW

    return U, V, W





    
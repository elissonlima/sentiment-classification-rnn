import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

def squared_error(y, y_pred):
    return float((y - y_pred) ** 2/2)

class RNN():

    bptt_truncate = 5 # backprop through time --> lasts 5 interations
    min_clip_val = -10
    max_clip_val = 10   

    def __init__(self, sequence_len, hidden_dim_len, output_len):

        # Crate RNN Model
        np.random.seed(12161)

        # weights from input to hidden layer
        self.U = np.random.uniform(0, 1, (hidden_dim_len, sequence_len))
        # weights from hidden to output layer
        self.V = np.random.uniform(0, 1, (output_len, hidden_dim_len))
        # recurrent weights for layer (RNN weigts)
        self.W = np.random.uniform(0, 1, (hidden_dim_len, hidden_dim_len))

        self.sequence_len = sequence_len


    def forward(self, x, prev_activation=None):
        '''
            Caculate activations of each timestep and returns activate of each
            timestep and matrixes of last timestep need for backpropagation

            Parameters:
                x : np.array
                    Input data point - expected shape:
                    (sequence_len, data_point_len)
                prev_activation : np.array, optional
                    Activation of last data point
                
            Returns:
                Y : np.array
                    Prediction output array
            
        '''
        timestep_activations = []

        if prev_activation is None:
            prev_activation = np.zeros(self.V.shape[::-1])

        for timestep in range(self.sequence_len):
            new_input = np.zeros(x.shape)
            new_input[timestep] = x[timestep]

            mul_u = np.dot(self.U, new_input)
            mul_w = np.dot(self.W, prev_activation)
            _sum = mul_w + mul_u
            activation = sigmoid(_sum)
            y_pred = np.dot(self.V, activation)

            timestep_activations.append({'activation':activation,
                            'prev_activation': prev_activation})
            prev_activation = activation
        
        return timestep_activations, mul_u, mul_w, y_pred

    def predict(self, X):
        '''
            Takes an input X and returns the output prediction Y_pred

            Parameters:
                X : np.array 
                    Input array - expected shape:
                    (num_samples, sequence_len, data_point_len)
            Returns:
                Y : np.array
                    Prediction output array
        '''
        predictions = []

        if len(X.shape) != 3:
            raise Exception("Not expected input shape. Expected 3 got %d" % \
                len(X.shape))

        for i in range(X.shape[0]):
            prev_activation = np.zeros(self.V.shape[::-1])

            for timestep in range(self.sequence_len):
                mul_u = np.dot(self.U, X[i])
                mul_w = np.dot(self.W, prev_activation)
                _sum = mul_u + mul_w

                activation = sigmoid(_sum)
                y_pred = np.dot(self.V, activation)
                prev_activation = activation
            
            predictions.append(y_pred)

        return np.array(predictions)


    def backprop(self, x, y_error, mul_u, mul_w, timestep_actv):
        """
            Execute truncate backpropagation through time

            Parameters: 
                x : np.array
                    Input for this data point
                y_error : float 
                    The error for the last layer
                mul_u : np.array 
                    The input value to the hidden layer
                mul_w : np.array
                    The input value to the hidden layer
                timestep_actv : list
                    The list of activations through timesteps
            Returns:
                dU : np.array
                    Gradients for weights from input to hidden layer
                dV : np.array
                    Gradients for weights from hidden to output layer
                dW : np.array
                    Gradients for recurrent weights for layer (RNN weigts)
        """     
        
        # First, set up the differential for each layer
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        # Then, set up the differential for each layer in the timestep
        dU_t = np.zeros(self.U.shape)
        dV_t = np.zeros(self.V.shape)
        dW_t = np.zeros(self.W.shape)

        # Then, we'll set up the differential for the truncated 
        # backpropagation through time
        dU_i = np.zeros(self.U.shape)
        dW_i = np.zeros(self.W.shape)

        # Finally, we'll set up the input weights of the hidden layer as the
        # most recent sum of the U and V matrix weight outputs and the
        # differential of the last layer
        _sum = mul_u + mul_w
        d_s_v = np.dot(self.V.T, y_error)

        # Defining a function to calculate the differential for the previous
        # activation of the hidden layer
        def get_previous_activation_differential(_sum, ds, W):
            d_sum = sigmoid_prime(_sum) * ds
            d_mul_w = d_sum * np.ones_like(ds)
            return np.dot(W.T, d_mul_w)

        # Next, we'll create the differential for this recurrent timestep by 
        # getting the dot product of the hidden weights and the timestep's 
        # previous activation. After this, we'll do the same step we do in the
        # forward pass by creating a new input for this recurrent timestep. 
        # This give us the differential for the input layer for this recurrent 
        # timestep. Finnaly, we'll increment the differential values for the 
        # hidden layer and the input layer with the differentials for the 
        # recurrent timestep.

        for timestep in range(self.sequence_len):

            dV_t = np.dot(y_error, timestep_actv[timestep]['activation'].T)            
            ds = d_s_v
            d_prev_activ = get_previous_activation_differential(
                _sum, ds, self.W
            )

            for _ in range(timestep-1, max(-1, timestep-self.bptt_truncate-1), -1):
                ds = d_s_v + d_prev_activ
                d_prev_activ = get_previous_activation_differential(
                    _sum, ds, self.W
                    )
                dW_i = np.dot(self.W, timestep_actv[timestep]['prev_activation'])

                new_input = np.zeros(x.shape)
                new_input[timestep] = x[timestep]
                dU_i = np.dot(self.U, new_input)

                dU_t += dU_i
                dW_t += dW_i

            dU += dU_t
            dV += dV_t
            dW += dW_t

            # take care of possible exploding gradients

            if dU.max() > self.max_clip_val:
                dU[dU > self.max_clip_val] = self.max_clip_val
            if dV.max() > self.max_clip_val:
                dV[dV > self.max_clip_val] = self.max_clip_val
            if dW.max() > self.max_clip_val:
                dW[dW > self.max_clip_val] = self.max_clip_val

            if dU.min() < self.min_clip_val:
                dU[dU < self.min_clip_val] = self.min_clip_val
            if dV.min() < self.min_clip_val:
                dV[dV < self.min_clip_val] = self.min_clip_val
            if dW.min() < self.min_clip_val:
                dW[dW < self.min_clip_val] = self.min_clip_val

        return dU, dV, dW

    def calculate_loss(self, X, Y):
        '''
            Calculate loss for entire dataset

            Parameters:
                X : np.array 
                    Input array - expected shape:
                    (num_samples, sequence_len, data_point_len)
                Y : np.array
                    Output array - expected shape:
                    (num_samples, data_point_len)
            Returns:
                loss : float
                    Loss value
                activation : np.array
                    Activation for last layer
        '''
        loss = 0.0

        for i in range(X.shape[0]):
            x, y = X[i], Y[i]
            timestep_activations, _, _, y_pred = self.forward(x)

            loss += squared_error(y, y_pred)
        
        return loss, timestep_activations[-1]['activation']
    
    def fit(self, X, Y, epochs, X_validation=None, Y_validation=None, lr = 0.0001):
        '''
            Train the model, adjusts the weights.
            
            Parameters:
                X : np.array 
                    Input array - expected shape:
                    (num_samples, sequence_len, data_point_len)
                Y : np.array
                    Output array - expected shape:
                    (num_samples, data_point_len)
            
        '''

        for epoch in range(epochs):
            loss, prev_activation = self.calculate_loss(X, Y)

            # check validation loss 
            if X_validation is not None and Y_validation is not None:
                val_loss, _ = self.calculate_loss(X_validation, Y_validation)
                print(f'Epoch: {epoch+1}, Loss: {loss}, Validation Loss: {val_loss}')
            else:
                print(f'Epoch: {epoch+1}, Loss: {loss}')

            for i in range(X.shape[0]):

                x, y = X[i], Y[i]
                timestep_actv, mul_u, mul_w, y_pred = self.forward(x, 
                                                prev_activation=prev_activation)

                error = y_pred - y 
                
                dU, dV, dW = self.backprop(x,error,mul_u,mul_w,timestep_actv)

                # update weights
                self.U -= lr * dU
                self.V -= lr * dV
                self.W -= lr * dW
            


        







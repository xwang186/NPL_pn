from keras import initializations
from keras.layers.recurrent import time_distributed_dense
from keras.activations import tanh, softmax
from keras.layers import LSTM
from keras.engine import InputSpec
import keras.backend as K
import numpy as np
class PointerLSTM(LSTM):
    def __init__(self, hidden_shape,*args,**kwargs):
        self.hidden_shape = hidden_shape
        self.input_length = []
        #self.clu=[]
        #for t in x:
         #   self.clu.append(km.encodeOH(km.kmeans(4,t),4))
        super(PointerLSTM, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(PointerLSTM, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]
        init = initializations.get('orthogonal')
        self.W1 = init((self.hidden_shape, 1))
        self.W2 = init((self.hidden_shape, 1))
        #self.W3 = init((10000,1))
        self.vt = init((input_shape[1], 1))
        self.trainable_weights += [self.W1, self.W2, self.vt]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        en_seq = x
        x_input = x[:, input_shape[1]-1, :]
        x_input = K.repeat(x_input, input_shape[1])
        initial_states = self.get_initial_states(x_input)

        constants = super(PointerLSTM, self).get_constants(x_input)
        constants.append(en_seq)
        preprocessed_input = self.preprocess_input(x_input)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             constants=constants,
                                             input_length=input_shape[1])

        return outputs

    def step(self, x_input, states):
        input_shape = self.input_spec[0].shape
        en_seq = states[-1]
        _, [h, c] = super(PointerLSTM, self).step(x_input, states[:-1])
        # vt*tanh(W1*e+W2*d)
        #p=K.variable(np.array(self.clu))
        #print("clu input shape: ",p)
        dec_seq = K.repeat(h, input_shape[1])
        Eij = time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = time_distributed_dense(dec_seq, self.W2, output_dim=1)
        #Cij = time_distributed_dense(p,self.W3,output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        #U = self.vt * tanh(Dij)
        U = K.squeeze(U, 2)


        # make probability tensor
        pointer = softmax(U)
        return pointer, [h, c]

    def get_output_shape_for(self, input_shape):
        # output shape is not affected by the attention component
        return (input_shape[0], input_shape[1], input_shape[1])

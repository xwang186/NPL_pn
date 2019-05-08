from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from PointerLSTM import PointerLSTM
import keras
import pickle
import tsp_data as tsp
import numpy as np
import matplotlib.pyplot as plt
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

nb_epochs = 15
learning_rate = 0.1
def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1
def runIt(hidden_size = 128,learning_r = 0.1,batch_s=30000,sameLen=True):
    print("preparing dataset...")
    t = tsp.Tsp()
    if sameLen:
        X, Y = t.next_batch(batch_s)
    else:
        X, Y = t.next_small_batch(batch_s)
    x_test, y_test = t.next_batch(1)
    learning_rate=learning_r
    YY = []
    for y in Y:
        YY.append(to_categorical(y))
    YY = np.asarray(YY)
    X_val, Y_val = t.next_batch(1000)
    YY_val = []
    for y in Y_val:
        YY_val.append(to_categorical(y))
    YY_val = np.asarray(YY_val)
    
    seq_len = 6
   
    

    print("building model...")
    main_input = Input(shape=(seq_len, 3), name='main_input')

    encoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
    decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

    model = Model(input=main_input, output=decoder)
    model.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = LossHistory()


    #X_val, Y_val = t.next_small_batch(1000)


    model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),history],validation_data=(X_val, YY_val))

    score_val = model.evaluate(X_val, YY_val, verbose=0)
    print("Let's do one test!")
    print("The Data I want to predict:")
    print(x_test)
    print("The result should be" ,y_test)
    print(model.predict(x_test))
    print("------")
    print(to_categorical(y_test))
    model.save_weights('model_weight_100.hdf5')


    #score_train=model.evaluate(X, YY, verbose=0)

    #print('Train score:', score_train[0])
    #print('Train accuracy:', score_train[1])
    print('Test score:', score_val[0])
    print('Test accuracy:', score_val[1])
    history.loss_plot('epoch')
#runIt(hidden_size=128,learning_r = 0.1,batch_s=30000)
#runIt(hidden_size=64,learning_r = 0.1,batch_s=30000)
#runIt(hidden_size=64,learning_r = 0.2,batch_s=10000,sameLen=True)
#runIt(hidden_size=64,learning_r = 0.2,batch_s=10000,sameLen=False)
#runIt(hidden_size=128,learning_r = 0.2,batch_s=10000,sameLen=True)
#runIt(hidden_size=128,learning_r = 0.2,batch_s=10000,sameLen=False)

runIt(hidden_size=128,learning_r = 0.14,batch_s=20000,sameLen=True)
runIt(hidden_size=128,learning_r = 0.14,batch_s=20000,sameLen=False)


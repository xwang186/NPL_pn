# NPL_pn

## How To run the project
Two folders are in this project. The one name without "modi" is the original pointer network implementation.
The one with modi is the archtecture I designed.
To run my project or the original one,
use python<3
keras=1.2.2
sudo pip install keras=1.2.2
(sudo if in docker) python run.py

## Description
Pointer-network is proven to have a good performance solving TSP(Trave Salesman Problem). I believe that the solution of a TSP problem should not be influenced by the order it feeds in. It should depend more on the location relative.
Suppose that we happen to have the distribution of the points before finding the path, we want to use the cluster distribution to help us better solve TSP.
In this task, I ran a preprocess, which is generating a cluster information using K-means. After that, I tried to plug in the cluster information into the original data and train the pointer network again.

## Conclusion
Seems that the new network uses less time to reach a reasonable training result. The validation shows that the result performs a little bit better than before. However, the network changes to larger and the number of to-be-trained parameters increases, so I think the result is not convincing enough.

## Training details
Structure:
        dec_seq = K.repeat(h, input_shape[1])
        Eij = time_distributed_dense(en_seq, self.W1, output_dim=1)
        Dij = time_distributed_dense(dec_seq, self.W2, output_dim=1)
        U = self.vt * tanh(Eij + Dij)
        U = K.squeeze(U, 2)
Parameter:
ncoder = LSTM(output_dim = hidden_size, return_sequences = True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, YY, nb_epoch=nb_epochs, batch_size=64,callbacks=[LearningRateScheduler(scheduler),])



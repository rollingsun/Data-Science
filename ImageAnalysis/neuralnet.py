import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.optimizers import Adagrad

dataset=np.loadtxt("train2.csv", delimiter=",")
X = dataset[:,1:785]
Y = dataset[:,0]

test=np.loadtxt("test2.csv", delimiter=",")
X_test=test[:,0:784]
batch_size = 100
nb_classes = 10
nb_epoch = 20
# the data, shuffled and split between train and test sets
X_train = X
Y_train = Y
#_test = X[30000:40000,]
#Y_test = Y[30000:40000,],
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test/=255
print(X_train.shape[0], 'train samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
#Y_test = np_utils.to_categorical(Y_test, nb_classes)


#Note-Changed to sigmoid activation and optimizer changed from adagrad. Previous accuracy ~97.69	
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adagrad(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1)

#score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

prediction=model.predict(X_test, batch_size=32, verbose=0)
prediction = np_utils.categorical_probas_to_classes(prediction)

np.savetxt('results.csv', 
           np.c_[range(1,len(test)+1),prediction], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')
import numpy as np 
import h5py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense,Conv3D,Dropout,MaxPooling3D,Flatten
import matplotlib.pyplot as plt



with h5py.File('yogaDataset3D.h5', 'r') as dataset:
     xtrain, xtest = dataset['X_train'][:], dataset['X_test'][:]
     ytrain, ytest = dataset['y_train'][:], dataset['y_test'][:]
     xtrain = np.array(xtrain)
     xtest = np.array(xtest)
print('train shape:', xtrain.shape)
print('train shape:', xtrain[0].shape)
print('train shape:', xtrain)
print('test shape:', xtest.shape)
xtrain = xtrain.reshape(xtrain.shape[0], 6, 4, 3, 1)
xtest = xtest.reshape(xtest.shape[0], 6, 4, 3, 1)
ytrain, ytest = to_categorical(ytrain, 28), to_categorical(ytest,28)


m= Sequential()
m.add(Conv3D(32,(2,2,2),activation='relu',input_shape=xtrain[0].shape,bias_initializer=Constant(0.01),padding="SAME"))
m.add(Conv3D(32,(2,2,2),activation='relu',bias_initializer=Constant(0.01)))
m.add(MaxPooling3D((1,1,1)))
m.add(Conv3D(64,(2,2,2),activation='relu'))
m.add(Conv3D(64,(3,1,1),activation='relu'))
m.add(MaxPooling3D((1,1,1)))
m.add(Dropout(0.6))

m.add(Flatten())

m.add(Dense(256,'relu'))
m.add(Dropout(0.7))
m.add(Dense(128,'relu'))
m.add(Dropout(0.5))
m.add(Dense(28,'softmax'))
m.summary()

#m.compile(Adam(0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
m.compile(Adam(0.001),'categorical_crossentropy',['accuracy'])
f=m.fit(xtrain,ytrain,epochs=100,batch_size=8,verbose=1,validation_data=(xtest,ytest))
# ,callbacks=[EarlyStopping(patience=15)]

accuracy_train = f.history['accuracy']
accuracy_val = f.history['val_accuracy']
epochs = range(1,101)
plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
plt.plot(epochs,accuracy_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


print("done1")
m.evaluate(xtest,ytest)
print("done2")
m.evaluate(xtrain,ytrain)

m.save("YogaPrediction3D.h5")
print("doneFinal")






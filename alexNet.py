# (1) Importing dependency
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1000)

# (2) Get Data
##import tflearn.datasets.oxflower17 as oxflower17
##x, y = oxflower17.load_data(one_hot=True)

train_df =pd.read_csv(r'shuffled training.csv')
test_df = pd.read_csv(r'shuffled testing.csv')

#print(train_df.head())

train_data = np.array(train_df,dtype='float32')
test_data = np.array(test_df,dtype='float32')


x_train = train_data[ : , 1:]/255
y_train = train_data[:,0]

x_test =test_data[:,1:] /255
y_test = test_data[:,0]

##x_train,x_validate,y_train,y_validate = train_test_split(
##    x_train,y_train,test_size=0.2,random_state=12345,
##
##    )
##
###image = x_train[100, :].reshape((28,28))
##
im_rows = 28
im_cols = 28
##batch_size = 32
im_shape = (im_rows,im_cols,1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
##x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)
##

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
##print('x_validate shape: {}'.format(x_validate.shape))


# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(28,28,1),kernel_size=(7,7),\
 strides=(2,2), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(1,1), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(28,28,1)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(284))
model.add(Activation('softmax'))

model.summary()

# (4) Compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',\
 metrics=['accuracy'])

# (5) Train
history = model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=1, \
validation_split=0.2, shuffle=True)



score = model.evaluate(x_test, y_test, verbose=0)
print('test loss: {:.4f}'.format(score[0]))
print(' test acc: {:.4f}'.format(score[1]))
    
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

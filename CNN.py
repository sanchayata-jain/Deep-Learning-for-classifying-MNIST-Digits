
import numpy as np 
import tensorflow as tf
from keras import layers
import mnist 
import pandas as pd

#USING MNIST
#MNIST training images stored in train_images, size = 60000x28x28
train_images = mnist.train_images() 
#MNIST training labels stored in train_labels, size = 60000x1                     
train_labels = (mnist.train_labels()).reshape((60000,1)) 
#MNIST test images stored in test_images size = 10000x28x28
val_images = mnist.test_images()    
#MNIST test images stored in test_labels, size = 10000x1                    
validation_labels = mnist.test_labels().reshape((10000,1))     

# Pre-processing images
#def pre_processing_images(image):
norm_train_img = train_images/255 #Normalising images
norm_val_img = val_images/255 
padded_tr_img = np.expand_dims(np.pad(norm_train_img,
                                      ((0,0),
                                       (2,2),
                                       (2,2)),
                                      'constant',
                                      constant_values = (0, 0)),
                                       3)
padded_val_img = np.expand_dims(np.pad(norm_val_img,
                                      ((0,0),
                                       (2,2),
                                       (2,2)),
                                      'constant',
                                      constant_values = (0, 0)),
                                       3)


labels = np.transpose(train_labels)
shape = (labels.size, labels.max()+1)
labels_oneC = np.zeros(shape)
label_rows = np.arange(labels.size)
labels_oneC[label_rows, labels] = 1

val_labels = np.transpose(validation_labels)
valshape = (val_labels.size, val_labels.max()+1)
val_labels_oneC = np.zeros(valshape)
val_label_rows = np.arange(val_labels.size)
val_labels_oneC[val_label_rows, val_labels] = 1

model = tf.keras.models.Sequential([
    
    layers.Conv2D(6, kernel_size = (5,5),
                  activation = 'tanh',
                  input_shape = (32,32,1)),
    
    layers.AveragePooling2D(2,2), 
    
    layers.Conv2D(16, kernel_size = (5,5), activation = 'tanh'),
    layers.AveragePooling2D(2,2),
    
    layers.Flatten(),
    
    layers.Dense(20, activation = 'tanh'),
    # softmax used because there are more than 2 classes
    layers.Dense(10, activation = 'softmax')   
    
])

model.summary()

epochs_num = 30
model.compile(optimizer = 'SGD', loss = 'MSE', metrics = ['accuracy'])
history = model.fit(padded_tr_img, labels_oneC, epochs=epochs_num, verbose=1,
                    validation_data=(padded_val_img, val_labels_oneC))

#history object has the attribute history which is a dictionary 
#of values of accuracy, etc
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(train_acc)) 

total_data = (np.vstack((train_loss, train_acc, val_loss, val_acc))).T
headings = ['Training Loss',
            'Training Accuracy',
            'Validation Loss',
            'Validation Accuracy']
df = pd.DataFrame(total_data,
                  index = range(1,len(train_acc)+1),
                  columns = headings)
df.to_csv('Enter Path Here')

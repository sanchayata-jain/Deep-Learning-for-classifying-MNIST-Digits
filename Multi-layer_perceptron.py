"""
Created 01/12/2020
@author: Sanchayata
Multilayer Perceptron
This program classifies handwritten digits from 0 to 9 using the MNIST
data set. The network has one hidden layer.
"""
# Import Libraries and Dataset
import numpy as np #to help create matricies
import pandas as pd #to export training and testing accuracy to a csv file
import mnist #The dataset

# Getting MNIST training and test images and labels
def importing_MNIST():
    #train images size = 60000x28x28
    train_images = mnist.train_images()     
    #train labels size = 60000x1                 
    train_labels = (mnist.train_labels()).reshape((60000,1))
    #test images size = 10000x28x28
    validation_images = mnist.test_images()                 
    #test labels size = 10000x1       
    validation_labels = mnist.test_labels().reshape((10000,1))     
    
    return train_images, train_labels, validation_images, validation_labels
    
# Pre-processing images 
# returns flattened images of size 60000x784 for train and 10000x784 for test
def pre_processing_images(image):
    N = image.shape[0] #Number of training/validation examples
    flat_image = np.transpose(image.reshape(N,-1))/255
    
    return flat_image

# Initalise weights and bias for each perceptron 
def initalisation(wr, wc, bc, m):
    weight = np.random.rand(wr, wc)*m
    bias = np.random.rand(1, bc)*m
    
    return weight, bias

# Forward Propagation 
def forward_prop(w, b, x):
    z = np.dot(x.T, w) + b
    y = 1/(1 + np.exp(-z)) 
    
    return y 

# Calculate training and test accuracy 
def accuracy_calc(y, d):
    N = d.shape[0]
    y_max = (np.argmax(y, axis = 1)).reshape((N, 1))
    compare = y_max == d
    correct = float(np.sum(compare))
    accuracy = (correct/N)*100 
    
    return accuracy 

# 1-of-C coding scheme (only used on training labels)
def one_of_C(labels):
    labels = np.transpose(labels) #labels.T = 1x60000
    shape = (labels.size, labels.max() + 1)
    labels_oneC = np.zeros(shape)
    label_rows = np.arange(labels.size)
    labels_oneC[label_rows, labels] = 1
    
    return labels_oneC

# Weight and Bias update equations
def weight_bias_update(w1, w2, b1, b2, a, d, y1, y2, x):
    w_updated2 = w2 + a*np.dot(y1.T, (d - y2)*y2*(1 - y2)) #output
    b_updated2 = b2 + a*np.sum((d - y2)*y2*(1 - y2), axis=0, keepdims=True)
    
    w_updated1 = w1 + a*np.dot(x, np.dot((d - y2)*y2*(1 - y2),
                                         w2.T)*y1*(1 - y1))
    b_updated1 = b1 + a*np.sum(np.dot((d - y2)*y2*(1 - y2), w2.T)*y1*(1 - y1),
                               axis = 0, keepdims = True)
    
    return w_updated1, b_updated1, w_updated2, b_updated2

# main function
def main():
    train_img, train_lab, val_img, val_lab = importing_MNIST()
    flat_tr = pre_processing_images(train_img) #training images
    flat_val = pre_processing_images(val_img) #test images
    d = one_of_C(train_lab) #one of c encoding of training labels
    
    alpha = 0.1 #learning rate
    multiplier = 1e-3
    epochs = 20
    n_in = flat_tr.shape[0]
    n_h = 20 #number of hidden perceptrons
    n_out = d.shape[1] #number of output perceptrons
    #creating an array to store training data in
    training_acc_arr = np.zeros((epochs,1)) 
    #creating an array to store validation data in
    validation_acc_arr = np.zeros((epochs, 1)) 
    
    w1, b1 = initalisation(n_in, n_h, n_h, multiplier) #hidden layer
    w2, b2 = initalisation(n_h, n_out, n_out, multiplier) #output layer
    
    for i in range(epochs):
        arr = np.random.permutation(60000)
        #2nd loop needed for stochasitc gradient descent to be implemented 
        for j in arr:   
            example = flat_tr[:,j].reshape((flat_tr.shape[0], 1))
            label = d[j,:].reshape((1, d.shape[1]))
            
            y_train1 = forward_prop(w1, b1, example) #output of hidden layer 
            y_train2 = forward_prop(w2, b2, y_train1.T)#output of output layer 
            w1, b1, w2, b2  = weight_bias_update(w1,
                                                 w2,
                                                 b1,
                                                 b2,
                                                 alpha,
                                                 label,
                                                 y_train1,
                                                 y_train2,
                                                 example)
        
        #To calculate training and validation accuracy
        y_train1 = forward_prop(w1, b1, flat_tr)
        y_train2 = forward_prop(w2, b2, y_train1.T)  
        y_validation1 = forward_prop(w1, b1, flat_val)
        y_validation2 = forward_prop(w2, b2, y_validation1.T)
        
        training_acc = accuracy_calc(y_train2, train_lab)
        training_acc_arr[i] = training_acc  #array used to export data to csv
        validation_acc= accuracy_calc(y_validation2, val_lab)
        validation_acc_arr[i] = validation_acc 
        
        #Priniting training and validation accuracy 
        print("training accuracy: "+str(training_acc))
        print("test accuracy: "+str(validation_acc))
        print("\n")

    
    # Exporting data to CSV file
    total_data = np.hstack((training_acc_arr, validation_acc_arr))
    headings = ['Training Accuracy (%)', 'Validation Accuracy (%)']
    df1 = pd.DataFrame(total_data,
                       index = range(1,training_acc_arr.shape[0]+1),
                       columns = headings)
    pd.DataFrame(df1).to_csv('Enter Path Here')
        
if __name__ == "__main__":
    main()

    
    
    
    
    
    
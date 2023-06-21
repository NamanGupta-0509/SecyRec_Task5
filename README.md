
## PClub Secretary Recruitment Tasks 2023


My submission for Task-5 : Machine Learning


 
    Naman Gupta
    - Roll no - 220686
    - Email - namangupta22@iitk.ac.in / namangupta0509@gmail.com
    - Phone no - 7678600231



GitHub Repo link: https://github.com/NamanGupta-0509/SecyRec_Task5/
## The Problem
Implement a feedforward neural network. This network will be trained and tested using the cifar-10. Specifically, given an input image (32x 32 pixels) from the cifar-10 dataset, the network will be trained to classify the image into 1 of 10 classes. Note, this neural network has to be implemented from scratch.

##

**NOTE** - Consider the file **'model_updated.ipynb'** as the final code and **'results.csv'** as the predicted output to the test csv file of CIFAR-10 dataset.


**NOTE** - I was unable to train the model locally on my machine, so I used kaggle and thus uploaded .ipynb file. Also as it could not be accessed via the command line, I added a separate function and wrote all the variables together. I will be  thankful for you accepting this. Moreover, I have run the program once and the results can be seen along the code as well.
## Overview of Code



I tried to proceed using classes for 'Layer' and 'NeuralNetwork' so that there remains flexibility of adding new layers and change their parameters as well.

Here just by calling the train function, we will be able to train the model on CIFAR-10 dataset.

```
lr = 0.2
momentum = 1
num_hidden = 3
sizes = (100,100,50)
activation = ('tanh', 'sigmoid', 'sigmoid')
loss = 'sce'
batch_size = 20
anneal = True
epochs = 500
# X_train, y_train, X_test, y_test already given

ann = NeuralNetwork()
def train(lr, momentum, num_hidden, sizes, activation, loss, batch_size, anneal, epochs, X_train, y_train, X_test, y_test):
    layer1 = Layer(3*32*32, sizes[0], activation[0])
    ann.add_layer(layer1)
    layer1.input = X_train
    for i in range(1, num_hidden):
        ann.add_layer(Layer(sizes[i-1], sizes[i], activation[i]))
    ann.add_layer(Layer(sizes[-1], 10, 'softmax'))
                  
    ann.forward_prop(X_train)
    ann.backward_prop(lr, y_train)

    ann.gradient_descent(X_train, y_train, lr=lr, iterations=epochs)
```

The code I've written is from stratch. I defined a class 'Layer' and a class 'NeuralNetwork' and built all the functions from there. The structure of ann is highly customizable with 'ReLU', 'Sigmoid', 'Tanh' and 'Softmax' as activation functions. The number of hidden layers and nodes can also be specified.

When I tested, the model gave best results (also considering time to train) when I used 3 hidden layers of 100, 100 and 50 nodes, with 'tanh' and 'sigmoid' activation functions.
## Visualize It

Some predictions were - 

Label - [Automobile] || Prediction - [Automobile]

Therefore, Correct!

![image](https://github.com/NamanGupta-0509/SecyRec_Task5/assets/66472692/de08a19f-ddcd-4bdd-a410-5c9e3120dc7d)



Label - [Automobile] || Prediction - [Automobile]

Sadly, Incorrect :(

![image](https://github.com/NamanGupta-0509/SecyRec_Task5/assets/66472692/86ca3a2a-4a85-4ab6-88f9-15135e8004fe)


## Python Code

Find the kaggle file here - 

model_updated.ipynb : https://www.kaggle.com/namangg0509/model-updated

model.ipynb : https://www.kaggle.com/namangg0509/namang-220686-task5

## Final Results
```
Accuracy is close to ~ 30 %
```
The accuracy is quite low, but it is justified as we are using a very basic ann. The task would have been more accurate if we used cnn or another advanced nn.
But as the accuracy on testing data does not deviate much from accuracy on training data, we can atleast say that the model is not over-fitted. 
## Thanks:)

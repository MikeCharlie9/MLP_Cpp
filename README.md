# MLP_Cpp

#### This is a C++ program for Multi-Layer Perceptron network


---

##### Introduction

The program can conduct training and test for MNIST which is a data set of handwritten numbers.

Detail information about MNIST can be found in the website: 
[MNIST](https://yann.lecun.com/exdb/mnist/)

The program is written in C++. No machining learning framework is used in the program to make sure that you can run the program in the simplest configuration.


---
##### File list
- NNmain.cpp
- NNclass.cpp
- NNpredef.h

NNmain.cpp is the file where the main function exists.

NNclass.cpp is the class defined to construct the network. If you have read the code, you can call the methods from this class.

NNpredef.h is the head file of the program. Some important parameters are defined here. You can change the structure of the network and the learning rate.

If you have read the code, you can modify anywhere as you like!


---
##### Run the program
First, you need to compile the source code. You may need a C++ compiler, such as MinGW. And then, the execution file is generated and you can run it.
Using the command like :

```
g++ NNmain.cpp -o NNmain.exe
NNmain.exe
```

On a normal PC(in 2019), it takes about 5 minutes to run a network with one hidden layer. Don't worry too much and you can sit down and have a coffee. You will see the middle result in each epoch.

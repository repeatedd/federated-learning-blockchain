# Federated Learning with Blockain
A simple application that uses Blockchain to demonstrate federated learning.

## What is Federated Learning?
<img src="https://1.bp.blogspot.com/-K65Ed68KGXk/WOa9jaRWC6I/AAAAAAAABsM/gglycD_anuQSp-i67fxER1FOlVTulvV2gCLcB/s640/FederatedLearning_FinalFiles_Flow%2BChart1.png"/>

Machine learning models trained on sensitive real-world data promise improvements to everything
from medical screening to disease outbreak discovery. And the widespread use of mobile devices
means even richer and more sensitive data is becoming available. However, traditional machine
learning involves a data pipeline that uses a central server(on-premise or cloud) that hosts the trained
model to make predictions. Distributed Machine Learning (FL) in contrast, is an approach that
downloads the current model and computes an updated model at the device itself (also known as edge
computing) using local data. Federated learning (FL) is a machine learning setting where many clients
(e.g. mobile devices or whole organizations) collaboratively train a model under the orchestration of a
central server (e.g. service provider) while keeping the training data decentralized.
Most of the previous research-work in federated learning focuses on the transfer and aggregation of the
gradients for learning on linear models and very less work is available on non-linear models. In this
project, we explore a secure decentralized learning model using neural networks. The motivation for
the same came from blockchain, preserving the user identities without a trusted central server and the
hidden layers in a neural network, able to add non-linearity to the model. Analogous to the transfer of
gradients in the federated learning system which requires a lot of network communication, we explore
the possibility of broadcasting weights to the blockchain system.
The goals of this work are to highlight research problems that are of significant theoretical and practical
interest and to encourage research on problems that could have significant real-world impact.

[blog](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

## Setup
Make sure you have python3, pip setup on your machine.

## Distributed Linear Regression Model

> ***Working Code Dir:*** Federated(linear Regression)
>
> ***Install librabries*** 
> > - pygad
> > - numpy
> > - pandas
> > - pickle
>
> ***Running of the Code***
> > - Open 3 terminals
> > - In first terminal, run ***python server.py***
> > - In second terminal, run ***python client1.py***
> > - In third terminal, run ***python client2.py*** 

## Distributed Blockchain Model

> ***Working Code Dir:*** Federated(linear regression + blockchain) 
>
> ***Structure of the block***
>> Block -
>> -    index
>> -    client_model
>> -    server_model
>> -    cli - [“cli1”, “cli2”]
>> -    timestamp
>> -    previous_hash
>> -    nonce
>
> ***Hashing of the Block***
>> Hash -
>>  -   index
>>  -   client_weights
>>  -   client_biases
>>  -   cli
>>  -   timestamp
>>  -   previous_hash
>>  -   nonce
> 
> ***Running of the Code***
> > - Open 3 terminals
> > - In first terminal, run ***python server.py***
> > - In second terminal, run ***python client1.py***
> > - In third terminal, run ***python client2.py*** 
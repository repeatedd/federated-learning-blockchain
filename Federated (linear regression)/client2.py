import socket
import pickle
import numpy

import pygad
import pygad.nn
import pygad.gann
import pandas
import math
import backprop as bp


# Data Input
df = pandas.read_csv('data.csv')

data = df[int(len(df)/2):]

X = data.drop('charges', axis=1)
y = data['charges']
y = numpy.array(y)
y = y.reshape((len(y), 1))


# Preparing the NumPy array of the inputs.
data_inputs = numpy.array(X)
# print("Shape of input",data_inputs.shape)

# Preparing the NumPy array of the outputs.
data_outputs = numpy.array(y)

data_inputs = data_inputs.T
data_outputs = data_outputs.T

mean = numpy.mean(data_inputs, axis = 1, keepdims=True)
std_dev = numpy.std(data_inputs, axis = 1, keepdims=True)
data_inputs = (data_inputs - mean)/std_dev


def recv(soc, buffer_size=1024, recv_timeout=10):
    received_data = b""
    while str(received_data)[-18:-7] != '-----------':
        try:
            soc.settimeout(recv_timeout)
            received_data += soc.recv(buffer_size)
        except socket.timeout:
            print("A socket.timeout exception occurred because the server did not send any data for {recv_timeout} seconds.".format(recv_timeout=recv_timeout))
            return None, 0
        except BaseException as e:
            return None, 0
            print("An error occurred while receiving data from the server {msg}.".format(msg=e))

    try:
        print("All data ({data_len} bytes).".format(data_len=len(received_data)))
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
        return None, 0

    return received_data, 1

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

try:
    soc.connect(("localhost", 10000))
    print("Successful Connection to the Server.\n")
except BaseException as e:
    print("Error Connecting to the Server: {msg}".format(msg=e))
    soc.close()
    print("Socket Closed.")

subject = "echo"
NN_model = None

while True:
    data = {"subject": subject, "data": NN_model, "mark":"-----------"}
    data_byte = pickle.dumps(data)
    print("data sent to server {}".format(len(data_byte)))
    
    print("Sending the Model to the Server.\n")
    soc.sendall(data_byte)
    
    print("Receiving Reply from the Server.")
    received_data, status = recv(soc=soc, 
                                 buffer_size=1024, 
                                 recv_timeout=10)
    if status == 0:
        print("Nothing Received from the Server.")
        break
    else:
        print(received_data, end="\n\n")

    subject = received_data["subject"]
    if subject == "model":
        NN_model = received_data["data"]
    elif subject == "done":
        print("Model is trained.")
        break
    else:
        print("Unrecognized message type.")
        break

    NN_model.data = data_inputs
    NN_model.labels = data_outputs

    history = NN_model.train(1000)
    prediction = NN_model.layers[-1].a
    error = NN_model.calc_accuracy(data_inputs, data_outputs, "RMSE")

    print("Error from model(RMSE) {error}".format(error = error))    

    subject = "model"

soc.close()
print("Socket Closed.\n")
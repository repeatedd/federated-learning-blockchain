import socket
import pickle
import numpy
import json

import pygad
import pygad.nn
import pygad.gann
import pandas
import math
import backprop as bp
import blockchain as bl


# Data Input

df = pandas.read_csv('data.csv')

data = df[:int(len(df)/2)]
cli = "cli_1"

X = data.drop('charges', axis=1)
y = data['charges']
y = numpy.array(y)
y = y.reshape((len(y), 1))

blockchain = bl.Blockchain()
blockchain.create_genesis_block()
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
        # print(str(received_data)[-18:-7])
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
chain = None

while True:
    data = {"subject": subject, "data": chain, "mark":"-----------"}
    data_byte = pickle.dumps(data)
    print("data sent to server {}".format(len(data_byte)))
    
    # for checking logs
    # f = open("logs.cli","a")
    # f.write(str(data_byte))
    # f.close()
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
        # NN_model = received_data["data"]
        chain = received_data["data"]
        print("Length of chain", len(blockchain.chain))
        last_block = blockchain.chain[-1]
        print("hash of last block client", last_block.hash)
        for k in chain:
            new_block = bl.Block(index=k.index,
                                 cli_model=k.cli_model,
                                 fin_model=k.fin_model,
                                 timestamp=k.timestamp,
                                 previous_hash=k.previous_hash,
                                 cli=k.cli,
                                 nonce=k.nonce)
            print("From ", k.cli)
            print("previous hash from server", k.previous_hash)
            proof = k.hash
            print("hash of this block", proof)
            blockchain = blockchain.add_block(new_block, proof)
            if not blockchain:
                raise Exception("The chain dump is tampered!!")
        # blockchain.add_blocks(chain)

        last_block = blockchain.chain[-1]
        # print(last_block.fin_model)
        NN_model = last_block.fin_model
        # print("Architecture of the model {}".format(NN_model.architecture))
        # print("Cost function the model {}".format(NN_model.cost_function))
    elif subject == "done":
        print("Model is trained.")
        break
    else:
        print("Unrecognized message type.")
        break

    # ga_instance = prepare_GA(GANN_instance)

    NN_model.data = data_inputs
    NN_model.labels = data_outputs

    history = NN_model.train(1000)
    # print(history)
    prediction = NN_model.layers[-1].a
    error = NN_model.calc_accuracy(data_inputs, data_outputs, "RMSE")

    # print("Predictions from model {predictions}".format(predictions = prediction))
    print("Error from model(RMSE) {error}".format(error = error))    
    # ga_instance.run()

    # ga_instance.plot_result()s

    subject = "model"
    chain = bl.Block(last_block.index+1,NN_model, 0, 0, last_block.hash, cli)

soc.close()
print("Socket Closed.\n")
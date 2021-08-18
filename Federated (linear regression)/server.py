import socket
import pickle
import threading
import time

import pygad
import pygad.nn
import pygad.gann
import numpy
import pandas
import backprop as bp

model = None
counter = 0

df = pandas.read_csv('data.csv')

X = df.drop('charges', axis=1)
y = df['charges']

y = numpy.array(y)
y = y.reshape((len(y), 1))

# Preparing the NumPy array of the inputs.
data_inputs = numpy.array(X)
# Preparing the NumPy array of the outputs.
data_outputs = y

data_inputs = data_inputs.T
data_outputs = data_outputs.T

mean = numpy.mean(data_inputs, axis = 1, keepdims=True)
std_dev = numpy.std(data_inputs, axis = 1, keepdims=True)
data_inputs = (data_inputs - mean)/std_dev


num_classes = 1
num_inputs = 12
sema = threading.Semaphore()

description = [{"num_nodes" : 12, "activation" : "relu"},
               {"num_nodes" : 1, "activation" : "relu"}]

NN_model = bp.NeuralNetwork(description,num_inputs,"mean_squared", data_inputs, data_outputs, learning_rat=0.001)

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.

                elif str(received_data)[-18:-7] == '-----------':
                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def model_averaging(self, model, other_model):
        for i in range(len(model.layers)):
            W_a = model.layers[i].W
            W_b = other_model.layers[i].W
            b_a = model.layers[i].b
            b_b = other_model.layers[i].b
            model.layers[i].W = (W_a + W_b)/2
            model.layers[i].b = (b_a + b_b)/2
        
        return model

    def reply(self, received_data):
        global NN_model, data_inputs, data_outputs, model, counter
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]

                if subject == "echo":
                    if model is None:
                        data = {"subject": "model", "data": NN_model, "mark": "-----------"}
                    else:
                        model.data = data_inputs
                        model.labels = data_outputs
                        model.forward_pass()
                        
                        predictions = model.layers[-1].a
                        error = model.calc_accuracy(data_inputs, data_outputs, "RMSE")

                        # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                        if error == 0:
                            data = {"subject": "done", "data": model, "mark": "-----------"}
                        else:
                            data = {"subject": "model", "data": model, "mark": "-----------"}

                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {msg}.\n".format(msg=e))
                elif subject == "model":
                    try:
                        best_model = received_data["data"]
                        if model is None:
                            model = best_model
                            print(model)
                        else:
                            model.data = data_inputs
                            model.labels = data_outputs
                            model.forward_pass()
                        
                            predictions = model.layers[-1].a

                            error = model.calc_accuracy(data_inputs, data_outputs, "RMSE")                            
                            # In case a client sent a model to the server despite that the model error is 0.0. In this case, no need to make changes in the model.
                            if error <= 0.15:
                                data = {"subject": "done", "data": None, "mark": "-----------"}
                                response = pickle.dumps(data)
                            else:
                                model = self.model_averaging(model, best_model)

                        model.data = data_inputs
                        model.labels = data_outputs
                        model.forward_pass()
                        
                        predictions = model.layers[-1].a

                        error = model.calc_accuracy(data_inputs, data_outputs, "MAE")
                        print("Error(RMSE) from {info} = {error}".format(error=error, info=self.client_info))
                        counter+=1
                        print("counter: ",counter)
                        if error >= 0.15:
                            data = {"subject": "model", "data": model, "mark": "-----------"}
                            print("sent", data)
                            response = pickle.dumps(data)
                            print("data_sent", len(response))
                            
                        else:
                            data = {"subject": "done", "data": None, "mark": "-----------"}
                            response = pickle.dumps(data)

                    except BaseException as e:
                        print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                else:
                    response = pickle.dumps("Response from the Server")
                            
                try:
                    self.connection.sendall(response)
                except BaseException as e:
                    print("Error Sending Data to the Client: {msg}.\n".format(msg=e))

            else:
                print("The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {d_keys}.".format(d_keys=received_data.keys()))
        else:
            print("A dictionary is expected to be received from the client but {d_type} received.".format(d_type=type(received_data)))

    def run(self):

        # This while loop allows the server to wait for the client to send data more than once within the same connection.
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            # Client data
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break

            sema.acquire()
            self.reply(received_data)
            sema.release()

soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
print("Socket Created.\n")

soc.bind(("localhost", 10000))
print("Socket Bound to IPv4 Address & Port Number.\n")

soc.listen(1)
print("Socket is Listening for Connections ....\n")

all_data = b""
while True:
    try:
        connection, client_info = soc.accept()
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info, 
                                     buffer_size=1024,
                                     recv_timeout=10)
        socket_thread.start()
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break

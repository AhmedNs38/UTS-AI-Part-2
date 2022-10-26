#AHMED NUR SIDIK 21091397038 2021B

#multiple perceptron/neuron batch and multiple layer

#inisialisasi numpy
import numpy as np

#inisialisasi variabel (input 10 ; batch 6 = matrix 6*10)
inputs = [[1.2, 2.0, 3.1, 3.4, 4.0, 5.6, 6.0, 7.3, 7.8, 8.5],
          [2.0, 3.0, 4.1, 4.6, 5.5, 6.0, 7.5, 1.3, 7.9, 3.0],
          [0.5, 1.3, 3.0, 5.1, 5.8, 6.1, 6.8, 4.3, 7.0, 7.2],
          [0.3, 2.1, 5.0, 5.3, 6.2, 6.5, 7.1, 7.7, 8.2, 8.8],
          [1.0, 5.6, 3.4, 2.5, 5.5, 6.4, 5.0, 7.5, 0.3, 1.5],
          [0.9, 7.1, 4.3, 4.9, 8.6, 5.0, 6.0, 5.6, 7.3, 2.3]]

#panjang weights = panjang inputs(10) ; jumlah weights = jumlah neuron(5)
weights1 = [[0.5, 1.5, 5.3, 7.3, 4.1, 7.0, 1.5, 4.3, 7.6, 6.1],
           [0.25, 3.6, 4.8, 5.2, 6.0, 6.4, 7.3, 7.5, 9.1, 9.7],
           [1.5, 0.3, 0.5, 2.3, 4.1, 5.0, 7.1, 7.3, 8.0, 9.0],
           [-0.25, 3.1, 4.0, 5.0, 6.0, 6.4, 7.8, 8.0, 8.1, 9.3],
           [0.5, 1.0, 2.1, 4.3, 4.4, 5.0, 6.3, 7.5, 8.2, 8.2]]

#jumlah biases pada layer1 adalah 5 neuron
biases1 = [8.0, 0.5, 1.2, 4.5, 7.4]

#panjang weights = neuron layer1(5) ; jumlah weights = jumlah neuron layer2(3)
weights2 = [[0.2, 1.3, 4.0, 5.1, 6.7],
            [2.1, 3.4, 4.3, 5.3, 6.0],
            [0.6, 1.7, 3.0, 4.7, 5.9]]

#jumlah biases pada layer2 3 neuron
biases2 = [1, 3, 6]

#command untuk menghitung layer1 menggunakan inputs, weights1, dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

#command untuk menghitung layer2 menggunakan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs)
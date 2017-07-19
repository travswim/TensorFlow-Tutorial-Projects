import tensorflow as tf

''' 
Input > weight > hidden  l1 (activation function) > weights > hidden l2
(activation function) > weights > output layer

compare output ot intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer.. SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch
'''

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# 'x' is input data, height x width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # This is a tensor (or array)
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                        'biasses': tf.Variable(tf.random_normal(n_nodes_hl1))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl2])),
                        'biasses': tf.Variable(tf.random_normal(n_nodes_hl2))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl3])),
                        'biasses': tf.Variable(tf.random_normal(n_nodes_hl3))}
    output_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                        'biasses': tf.Variable(tf.random_normal(n_nodes_hl1))}
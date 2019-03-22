import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

train_x = np.load('/home/souvik/Desktop/DEEP_learning/assignment_1/mnist/x_train.npy')
train_y = np.load('/home/souvik/Desktop/DEEP_learning/assignment_1/mnist/y_train.npy')
test_x = np.load('/home/souvik/Desktop/DEEP_learning/assignment_1/mnist/x_test.npy')
test_y = np.load('/home/souvik/Desktop/DEEP_learning/assignment_1/mnist/y_test.npy')
train = np.split(train_x,2)
test = np.split(test_x,2)
train_x = train[0]
#train_y = train[1]
test_x = test[0]
#test_y = test[1]
num_pixels =train_x.shape[1] * train_x.shape[2]
train_x = train_x.reshape(train_x.shape[0], num_pixels).astype('float32')
train_y = train_y.reshape(30000,2).astype('float32')
print(train_x.shape)
num_pixels_1 =test_x.shape[1] * test_x.shape[2]

test_x = test_x.reshape(test_x.shape[0], num_pixels_1).astype('float32')
test_y = test_y.reshape(5000,2).astype('float32')

# define the important parameters and variables to work with the tensors
learning_rate = 0.0003
training_epoch = 10
cost_history = np.empty(shape=[], dtype=float)
print(test_y.shape)
n_dim = 784
print("n_dim", n_dim)
n_class = 2
model_path = "/home/souvik/Desktop/DEEP_learning/assignment_1"

# define the number of hidden layers and neurons for each layer
n_hidden_1 = 128
n_hidden_2 = 128
#n_hidden_3 = 20
#n_hidden_4 = 20

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

# define the model
def multilayer_perceptrons(x, weights, biases):

    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # hidden layer with sigmoid activation
    #layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    #layer_3 = tf.nn.sigmoid(layer_3)

    # hidden layer with RELU activation
    #layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    #layer_4 = tf.nn.relu(layer_4)

    # output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# define the weights and the biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    #'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    #'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_class]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    #'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    #'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}


# initialize all the variable

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# call your model defined
y = multilayer_perceptrons(x, weights, biases)
#y = tf.reshape(y,[30000,2])
# define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(init)

# calculate the cost and the accuracy for each epoch

mse_history = []
accuracy_history = []

for epoch in range(training_epoch):
    sess.run(training_step, feed_dict={x:train_x, y_:train_y})
    cost = sess.run(cost_function, feed_dict={x:train_x, y_:train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    print('epoch:', epoch, '-', cost, "-MSE: ", mse_, "-Train Accuracy ", accuracy)

save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

# plot mse and accuracy graph

plt.plot(mse_history, 'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

# print the final accuracy

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))

# print the final mse

pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
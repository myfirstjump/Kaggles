import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data   # its deprecated
# from keras.datasets import mnist


'''
input > weights > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output

compare output to intended output > coss function (cross entropy)
optimizer > minimize cost (AdamOptimizer...SGD, Adagrad)
backpropagation

feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets('/tmp/data/', one_hot = True)
# mnist_train, mnist_test = mnist.load_data() # 想使用 keras 的 module，不過後面有使用

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_class = 10 # 10 classes, 0-9
batch_size = 100 # 在某些資料集，RAM可能不足以支持full-batch的資料量

x = tf.placeholder('float', [None, 784]) # input 的 shape 是optional輸入的
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weigths':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} # biases想像成，調整neural發火的變量

    hidden_2_layer = {'weigths':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
                      
    hidden_3_layer = {'weigths':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weigths':tf.Variable(tf.random_normal([n_nodes_hl3, n_class])),
                      'biases':tf.Variable(tf.random_normal([n_class]))}
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weigths']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weigths']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weigths']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.matmul(data, output_layer['weigths']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logit(prediction, y))

    # learning rate = 0.001 (default)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 5 # how many epochs
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in hm_epochs:
            epoch_loss = 0
            for _ in range(int(mnist.trian.num_examples/batch_size)): # 所有samples/batch大小  # mnist.train是內建的
                x, y = mnist.train.next_batch(batch_size) # mnist.train是內建的，其他資料集要自己寫這個部分
                _, c = sess.run([optimizer, cost], feed_dict = {x:x, y:y})
                epoch_loss += c
            print('Epoch', epoch, 'complete out of', hm_epochs, 'loss', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
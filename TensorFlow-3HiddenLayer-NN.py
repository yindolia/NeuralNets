import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

n_nodes_hl1= 500
n_nodes_hl2= 500
n_nodes_hl3= 500

num_class = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def NN_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
					 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, num_class])),
					  'biases': tf.Variable(tf.random_normal([num_class]))}


	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) , hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output


def train_NN(x):
	prediction = NN_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=prediction, logits= y))
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(n_epochs):
			epoch_loss =0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size)
				_ , c = sess.run([optimizer, cost], feed_dict = {x:x, y:y})
				epoch_loss += c

			print ('Epoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

		correct_answer = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct_answer, 'float'))

	#	print ('Accuracy': accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_NN(x)

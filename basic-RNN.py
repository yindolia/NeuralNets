import numpy as np
import theano
import theano.tensor as T
import time
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.ion()


mode = theano.Mode(linker='cvm') #runtime algo in c

class RNN:
	def __init__(self, n_u, n_h, n_y, activation, output_type,
				learning_rate, learning_rate_decay, L1_reg, L2_reg,
				initial_momentum, final_momentum, momentum_switchover,
				n_epochs):
		self.n_u = int(n_u)
		self.n_h = int(n_h)
		self.n_y = int(n_y)

		if activation == 'tanh':
			self.activation = T.tanh
		elif activation == 'sigmoid':
			self.activation == T.nnet.sigmoid
		elif activation == 'relu':
			self.activation == lambda x: x * (x>0) # T.maximum(x,0)
		else:
			raise NotImplementedError

		self.output_type = output_type
		self.learning_rate = float(learning_rate)
		self.learning_rate_decay= float(learning_rate_decay)
		self.L1_reg = float(L1_reg)
		self.L2_reg = float(L2_reg)
		self.initial_momentum = float(initial_momentum)
		self.final_momentum = float(final_momentum)
		self.momentum_switchover = int(momentum_switchover)
		self.n_epochs = int(n_epochs)

		self.x = T.matrix()

		#weights initialized from uniform distribution

		self.W_uh = theano.shared(value= np.asarray(
										np.random.uniform(
											size= (n_u,n_h), low= -.01, high = .01),
										dtype = theano.config.floatX),
								name = 'W_uh')

		self.W_hh = theano.shared(value = np.asarray(
										np.random.uniform(
											size= (n_h, n_h), low= .01, high = .01),
										dtype = theano.config.floatX),
								name = 'W_hh')

		self.W_hy = theano.shared(value = np.asarray(
										np.random.uniform(
											size= (n_h, n_y), low= .01, high = .01),
										dtype = theano.config.floatX),
								name = 'W_hy')


		# Hidden Layer initial layer units are set to zero

		self.h0 = theano.shared(value = np.zeros(
											(n_h, ), dtype = theano.config.floatX),
								name = 'h0')

		# Biases initialized to zero

		self.b_h = theano.shared(value = np.zeros(n_h, ), dtype = theano.config.floatX, name = 'b_h')
		self.b_y = theano.shared(value = np.zeros(n_h, ), dtype = theano.config.floatX, name = 'b_y')

		self.params = [self.W_uh, self.W_hy, self.W_hh, self.h0, self.b_h, self.b_y]

		self.updates = {}
		for param in self.params:
			self.updates[param] = theano.shared(value = np.zeros(param.get_value(borrow = True).shape, dtype = theano.config.floatX),
												name = 'updates')

		# h_t = sig(W_uh * u_t + W_hh *h_t-1 + b_h)
		# y_t = W_yh * h_t + b_y

		def recurrent_fn(u_t, h_t1):
			h_t = self.activation(T.dot(u_t, self.W_uh) + \
									T.dot(h_t1, self.W_hh) + \
									self.b_h)
			y_t = T.dot(h_t, self.W_hy) + self.b_y

			return h_t, y_t



		[self.h, self.y_pred], _ = theano.scan(recurrent_fn, sequences = self.x, output_info = [self.h0, None])
		#Iteration over Time in Tensor
		#scan updates returned not needed
			

		self.L1 = abs(self.W_uh.sum()) + abs(self.W_hh.sum()) + abs(self.W_hy.sum())

		self.L2_square = abs(self.W_uh**2).sum() + abs(self.W_hh**2).sum() + abs(self.W_hy**2).sum()

		# Loss Functions

		if self.output_type == 'real':
			self.y = T.matrix(name = 'y', dtype= theano.config.floatX)
			self.loss = lambda y: self.mse(y) 
			self.predict = theano.function(inputs= [self.x, ], outputs= self.y_pred, mode= mode)

		elif self.output_type == 'binary':
			self.y = T.matrix(name = 'y', dtype = 'int32')
			self.pCond_y_x = T.nnet.sigmoid(self.y_pred)
			self.y_out = T.round(self.pCond_y_x) # round to {0,1}
			self.loss = lambda y: self.nll_binary(y)
			self.predict_prob = theano.function(inputs= [self.x, ], outputs = self.pCond_y_x, mode = mode)
			self.predict = theano.function(inputs= [self.x, ], outputs = T.round(self.pCond_y_x), mode = mode)


		elif self.output_type == 'softmax':
			self.y = T.vector(name = 'y', dtype= 'int32')
			self.pCond_y_x = T.nnet.softmax(self.y_pred)
			self.y_out = T.argmax(self.pCond_y_x, axis= -1)
			self.loss = lambda  y: self.nll_multiclass(y)
			self.predict_prob = theano.function(inputs= [self.x, ], outputs = self.pCond_y_x, mode = mode)
			self.predict = theano.function(inputs= [self.x, ], outputs = self.y_out, mode = mode)

		else:
			raise NotImplementedError

		self.errors = []



	def mse(self, y):
		return T.mean((self.y_pred - y)** 2)


	def nll_binary(self,y):

		return T.mean(T.nnet.binary_crossentropy(self.pCond_y_x, y)) # negetive log likelihood


	def nll_multiclass(self, y):

		return -T.mean(T.log(self.pCond_y_x)[T.arange(y.shape[0]), y])



	def build_and_train(self, x_train, y_train, x_test = [], y_test = []):
		train_set_x = theano.shared(np.asarray(x_train, dtype = theano.config.floatX))
		train_set_y = theano.shared(np.asarray(y_train, dtype = theano.config.floatX))

		if output_type in ('binary', 'softmax'):
			train_set_y = T.cast(train_set_y, 'int32')

		print ('Build Model.....')

		index = T.lscalar('index')

		learn_rate = T.scalar('learn_rate', dtype= theano.config.floatX)
		momentum = T.scalar('momentum', dtype = theano.config.floatX)

		#Use cost for training, compute_train_error for watching

		cost = self.loss(self.y) + self.L1_reg * self.L1 + self.L2_reg * self.L2_square

		compute_train_error = theano.function(inputs = [index, ], outputs = self.loss(self.y),
												givens = {self.x: train_set_x[index], self.y: train_set_y[index]}, mode = mode)


		gparams = []


		
		# Gradient of cost wrt [self.W, self.W_in, self.W_out, self.h0, self.b_h, self.b_y] using BPTT
		

		for param in self.params:
			gparams.append(T.grad(cost, param))


		self.updates = {}
		for param, gparam in zip(self.params, gparams):
			weight_update = self.updates[param]
			upd = momentum*weight_update -lr *gparam
			updates[weight_update] = upd
			updates[param] = param +upd


		#TRAIN MODEL

		epoch = 0
		n_train = train_set_x.get_value(borrow = True).shape[0]

		while(epoch< self.n_epochs):
			epoch = epoch+1
			for ind in xrange(n_train):
				update_momentum = self.final_momentum if epoch > self.momentum_switchover \
													  else self.initial_momentum

				example_cost = train_model(ind, self.learning_rate, update_momentum)


			train_losses = [compute_train_error[i] for i in xrange(n_train)]
			training_losses_this = np.mean(train_losses)

			print ('epoch %i, trian loss %f''lr:%g' %(epoch, training_losses_this, self.learning_rate))

			self.learning_rate *= self.learning_rate_decay






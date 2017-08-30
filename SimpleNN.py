import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)


np.random.seed(0)
X,y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0],X[:,1], s = 40, c=y, cmap=plt.cm.Spectral)

#Logistic regression

clf1 = sklearn.linear_model.LogisticRegression()
clf1.fit(X,y)

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X,y)

def plot_decision_boundary(prediction_function):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() +0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() +0.5
    h = 0.01 
    #h for generating grid
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z= prediction_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap= plt.cm.Spectral)
    plt.scatter(X[:,0], X[:,1], c= y, cmap= plt.cm.Spectral)
    
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic")

num_examples= len(X)
nn_input_dim = 2
nn_output_dim = 2
epsilon = 0.01 #learning rate
reg_lambda = 0.01 #regularization

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1= X.dot(W1) +b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2)+ b2
    exp_scores = np.exp(z2)
    probs = exp_scores/ np.sum(exp_scores, axis=1, keepdims= True)
    
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    
    data_loss += reg_lambda/2* (np.sum(np.sqaure(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def predict(model, X):
    W1, b1, W1, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1= X.dot(W1) +b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2)+ b2
    exp_scores = np.exp(z2)
    probs = exp_scores/ np.sum(exp_scores, axis=1, keepdims= True)
    return np.argmax(probs, axis =1)
    
def build_model(nn_hdim, num_passes= 20000, print_loss = False):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

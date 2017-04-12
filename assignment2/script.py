import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys
def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    

    # IMPLEMENT THIS METHOD   
    d = np.shape(X)[1]
    k = int(np.max(y))

    means = np.zeros((d,k))
    for i in range(1, k+1):
        class_i = X[np.where(y==i)[0]]
        means[:,i-1] = np.mean(class_i, axis = 0)
              
    covmat = np.cov(X, rowvar = False)    
   

    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    d = np.shape(X)[1]
    k = int(np.max(y))
    
    means = np.zeros((d,k))
    covmats = []
    for i in range(1, k+1):
        class_i = X[np.where(y==i)[0]]
        means[:,i-1] = np.mean(class_i, axis = 0)
        covmats.append(np.cov(class_i, rowvar = False))
        

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    
    # IMPLEMENT THIS METHOD

    N = np.shape(Xtest)[0]
    d = np.shape(Xtest)[1]
    k = np.shape(means)[1]
    ypred = []
    inv_covmat = np.linalg.inv(covmat)  
    for x in Xtest:
        
        pred = []
        #P_x = 0
        for mu in range(k):         
            XminusMu = x-means[:,mu]            
            pred_i = 1/(np.power(2*np.pi, d/2)*np.power(np.linalg.det(covmat),1/2))* np.exp(-.5*((XminusMu.T).dot(inv_covmat).dot(XminusMu)))
            pred.append(pred_i)
            continue
        k_max = np.where(pred == np.max(pred))[0]
        ypred.append(k_max+1)
    ypred = np.array(ypred)
    acc = np.sum(ypred == ytest) / N


    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    N = np.shape(Xtest)[0]
    d = np.shape(Xtest)[1]
    k = np.shape(means)[1]
    
    ypred = []
    for x in Xtest:
        inv_covmats = np.linalg.inv(covmats)  
        pred = []
        for mu in range(k):         
            XminusMu = x-means[:,mu]            
            pred_i = 1/(np.power(2*np.pi, d/2)*np.power(np.linalg.det(covmats[mu]),1/2))* np.exp(-.5*((XminusMu.T).dot(inv_covmats[mu]).dot(XminusMu)))
            pred.append(pred_i)
            continue
        k_select = np.where(pred == np.max(pred))[0] + 1
        ypred.append(k_select)
    ypred = np.array(ypred)
    acc = np.sum(ypred == ytest) / N

    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 

    	
    # IMPLEMENT THIS METHOD       
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)                                         

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1       
    
    d = np.shape(X)[1]                                                         
    I_d = np.identity(d)


    # IMPLEMENT THIS METHOD    
    w = np.linalg.inv(lambd*I_d + X.T.dot(X)).dot(X.T).dot(y)                                               

    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    N = np.shape(Xtest)[0]
    
    # IMPLEMENT THIS METHOD

    mse = (1/N)*(ytest - Xtest.dot(w)).T.dot(ytest - Xtest.dot(w))

    return mse



def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  


    # IMPLEMENT THIS METHOD   
    w = np.reshape(w,(65,1))
    error = (0.5*(y - X.dot(w)).T.dot(y - X.dot(w))) + (0.5 * lambd * w.T.dot(w))
    
    error_grad = X.T.dot(X.dot(w) - y) + (lambd * w)
    #error_grad = np.dot(np.transpose(X),((np.dot(X,w)-y)))+(np.multiply(lambd,w))
    error_grad = error_grad.flatten()
#    error_grad = np.reshape(error_grad,(65,))
  
 
   
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
    
    N = np.shape(x)[0]
    # IMPLEMENT THIS METHOD

    Xd = np.zeros((N,p+1))
    for i in range(0, p+1):
        Xd[:,i-1] = np.power(x,i)

    return Xd

# Main script


# Problem 1
# load the sample data                                                                 

## Problem 1
## load the sample data                                                                 

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')


plt.show()

# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

mle_train = testOLERegression(w,X,y)
mle_i_train = testOLERegression(w_i,X_i,y)

print('MSE without intercept training data : '+str(mle_train))
print('MSE with intercept training data : '+str(mle_i_train))
print('MSE without intercept test data: '+str(mle))
print('MSE with intercept test data: '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')


fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(w_i)
plt.title('Magnitudes of Weights Learnt using OLE')
plt.subplot(1, 2, 2)
plt.plot(w_l)
plt.title('Magnitudes of Weights Learnt using Ridge Regression')


plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt =0.06
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

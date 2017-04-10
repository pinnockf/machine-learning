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
    X_1=X[np.where(y==1)[0],:]
    X_2=X[np.where(y==2)[0],:]
    X_3=X[np.where(y==3)[0],:]
    X_4=X[np.where(y==4)[0],:]
    X_5=X[np.where(y==5)[0],:]

    mu1=[np.mean(X_1[:,0]),np.mean(X_1[:,1])]
    mu2=[np.mean(X_2[:,0]),np.mean(X_2[:,1])]
    mu3=[np.mean(X_3[:,0]),np.mean(X_3[:,1])]
    mu4=[np.mean(X_4[:,0]),np.mean(X_4[:,1])]
    mu5=[np.mean(X_5[:,0]),np.mean(X_5[:,1])]

    MU=np.array([mu1,mu2,mu3,mu4,mu5])
    means=np.transpose(MU)

    covmat=np.cov(X.T)
    
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
    X_1=X[np.where(y==1)[0],:]
    X_2=X[np.where(y==2)[0],:]
    X_3=X[np.where(y==3)[0],:]
    X_4=X[np.where(y==4)[0],:]
    X_5=X[np.where(y==5)[0],:]

    mu1=[np.mean(X_1[:,0]),np.mean(X_1[:,1])]
    mu2=[np.mean(X_2[:,0]),np.mean(X_2[:,1])]
    mu3=[np.mean(X_3[:,0]),np.mean(X_3[:,1])]
    mu4=[np.mean(X_4[:,0]),np.mean(X_4[:,1])]
    mu5=[np.mean(X_5[:,0]),np.mean(X_5[:,1])]

    MU=np.array([mu1,mu2,mu3,mu4,mu5])
    means=np.transpose(MU)
    
    cov1=np.cov((X_1.T))
    cov2=np.cov((X_2.T))
    cov3=np.cov((X_3.T))
    cov4=np.cov((X_4.T))
    cov5=np.cov((X_5.T))

    covmats=np.array([cov1,cov2,cov3,cov4,cov5])
  
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    ypred=[]
    for i in Xtest:
        pk=0
        for j in range(5):
            newp=(1/((sqrt(2*pi))*(np.linalg.det(covmat)**0.5)))*np.exp(-0.5*(np.dot(np.dot((i-means[:,j]).T,np.linalg.inv(covmat)),(i-means[:,j]))))
            if newp>pk:
                pk=newp
                label=j+1

        
        ypred.append([float(label)])
    
    ypred=np.array(ypred)
    
    acc=100*np.mean((ypred == ytest).astype(float))
  
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    
    label=0
    ypredict=[]
    for i in Xtest:
        pk=0
        for j in range(5):
            newp=(1/((sqrt(2*pi))*(np.linalg.det(covmats[j])**0.5)))*np.exp(-0.5*(np.dot(np.dot((i-means[:,j]).T,np.linalg.inv(covmats[j])),(i-means[:,j]))))
            if newp>pk:
                pk=newp
                label=j+1
        
        ypredict.append([float(label)])
    ypredict=np.array(ypredict)
    
    acc=100*np.mean((ypredict == ytest).astype(float))
    return acc,ypredict

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD
    w=np.linalg.solve(np.dot(np.transpose(X),X),np.dot(np.transpose(X),y))
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    w=np.linalg.solve(np.dot(np.transpose(X),X)+lambd*np.eye(np.shape(X)[1],np.shape(X)[1]),np.dot(np.transpose(X),y))                                        

    
    # IMPLEMENT THIS METHOD                                                   
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD

    total=0
    for j in range(len(Xtest)):
        total+=((ytest[j]-sum(Xtest[j][i]*w[i] for i in range(len(w))))**2)
    rmse=(sqrt(total/len(Xtest)))

    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    # IMPLEMENT THIS METHOD
    
    w=np.reshape(w,(65,1))
  
    err=(y-np.dot(X,w))
    error=(sum(err*err)*0.5)+(0.5*lambd*(np.dot(np.transpose(w),w)))
    
    
    error_grad=np.dot(np.transpose(X),((np.dot(X,w)-y)))+(np.multiply(lambd,w))
    error_grad=np.reshape(error_grad,(65,))
    
    
    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    
        Xd=np.ones((len(x),p+1))

        for i in range(len(x)):
                for j in range(p+1):
                        Xd[i][j]=x[i]**j
        return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')       

### LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldapred = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
### QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdapred = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

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

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))


# Problem 3
k =101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
plt.show()


# Problem 4
k =101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100} # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))


for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
               
plt.show()

# Problem 5

pmax = 7
lambda_opt=lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))

for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()






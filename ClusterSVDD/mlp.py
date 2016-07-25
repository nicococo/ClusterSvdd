import numpy
import numpy as np

from sklearn.svm import SVR

#--------------------------------------------------------------------
# Hyperparameters
#--------------------------------------------------------------------
lr = 0.001 # learning rate


#--------------------------------------------------------------------
# Multilayer network
#--------------------------------------------------------------------
class Sequential:

    def __init__(self,modules): self.modules = modules

    def forward(self,X):
        for m in self.modules: X = m.forward(X)
        return X

    def backward(self,DY):
        for m in self.modules[::-1]: DY = m.backward(DY)
        return DY

    def update(self):
        for m in self.modules: m.update()

#--------------------------------------------------------------------
# Linear layer
#--------------------------------------------------------------------
class Linear:

    def __init__(self,m,n,last=False):
        self.m = m
        self.n = n

        self.W = numpy.random.uniform(-1/self.m**.5,1/self.m**.5,[m,n]).astype('float32')
        self.B = numpy.zeros([n]).astype('float32')
        if last: self.W *= 0

    def forward(self,X):
        self.X = X
        return numpy.dot(X,self.W)+self.B

    def backward(self,DY):

        DX = numpy.dot(DY,self.W.T)

        self.DW = (numpy.dot(self.X.T,DY))/ self.m**.5
        self.DB = (DY.sum(axis=0)) / self.m**.25

        return DX*(self.m**.5/self.n**.5)

    def update(self):
        self.W -= lr*self.DW
        self.B -= lr*self.DB

#--------------------------------------------------------------------
# Hyperbolic tangent layer
#--------------------------------------------------------------------
class Tanh:
    def __init__(self): pass
    def forward(self,X): self.Y = numpy.tanh(X); return self.Y
    def backward(self,DY): return  DY*(1-self.Y**2)

    def update(self): pass


#====================================================================
# Test
#====================================================================

# Prepare data
nbsamples=200
nbinputdims=100
nboutputdims=1

# Random regression task
X = numpy.random.normal(0,1,[nbsamples,nbinputdims])
T = numpy.random.normal(0,1,[nbsamples,nboutputdims])
T = numpy.random.normal(0,1,[nbsamples])

# Initialize network
nn = Sequential([
    Linear(nbinputdims,200),
    Tanh(),
    Linear(200,20),
    Tanh(),
    Linear(20,nboutputdims)
])

clf = SVR(C=1000.0, epsilon=0.0002)
clf.fit(X, T)
ypred = clf.predict(X)
print((ypred-T)**2).sum()

T = T[:,np.newaxis]

# Training
for t in range(1000):

    Y = nn.forward(X)
    nn.backward(Y-T)
    nn.update()

    if t % 100 == 0: print(t,((Y-T)**2).sum())


import numpy as np

X = np.asarray([[0.1, 0.2, 0.3],
                [0.5, 0.6, 0.7]])
Y = np.asarray([0.1, 0.3])

W = np.asarray([0.0, 0.0, 0.0])
B = 0.0

def f(w,b,x) : 
  return 1/(1 + np.exp(-(np.dot(x,w) + b)))

def grad_b(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*fx*(1-fx)

def grad_w(w,b,x,y):
  fx = f(w,b,x)
  return (fx-y)*fx*(1-fx)*x

#training
def grad_descent():
  
  w,b,eta,epochs = [-2.0,-2.0,-2.0], -2.0, 1.0,10000
  
  for r in range(epochs):
    
    dw, db = [0.0, 0.0, 0.0] , 0.0
    
    #accumulate dw db 
    for i in range(len(X)):
      dw += grad_w(w,b,X[i],Y[i])
      db += grad_b(w,b,X[i],Y[i])
      
    #update w and b
    w = w - eta*dw
    b = b - eta*db
  return w,b

W, B = grad_descent()

prediction = f(W,B,X)
prediction


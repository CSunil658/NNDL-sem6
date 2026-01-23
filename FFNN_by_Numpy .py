import numpy as np
x =np.array([0.5,0.6,0.8])
w = np.array([
    [170, 22],
    [160, 20],
    [175, 23]
])
b = np.array([0.01,0.02])
z = np.dot(x,w) +b 
print("output",z)
def sigmoid(x):
    return 1/(1+np.exp(-x))
a = sigmoid(z)
print("Activated ouput:",a)
w2 = np.array([[0.1], [0.2]])
b2 = np.array([0.05])

z2 = np.dot(a, w2) + b2
a2 = sigmoid(z2)
print("final output", a2)
y=1
loss = (y-a2)**2
print("loss", loss)
lr = 0.1
dloss_da2 = 2*(y-a2)
da2_dz2 = a2*(1-a2)
dz2_dw2 = a

grad_w2 = dloss_da2*da2_dz2*dz2_dw2
w2 = w2 - lr*grad_w2.reshape(2,1)
print("updated w2", w2)

output:

output [321.01  41.42]
Activated ouput: [1. 1.]
final output [0.58661758]
loss [0.17088503]
updated w2 [[0.07995117]
 [0.17995117]]

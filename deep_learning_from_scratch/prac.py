import numpy as np

X = np.random.rand(2) #입력
W = np.random.rand(2,3) #가중치
B = np.random.rand(3) #편향

Y= np.dot(X, W) + B
print(Y)


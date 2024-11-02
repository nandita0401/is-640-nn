from engine import Value
from nn import MLP

x = [2.0, 3.0, -1.0]
n = MLP(3, [4,4,1])
n(x)

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0] #desired targets

for k in range(10):

    #forward pass
    ypred = [n(x) for  x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    
    #backward pass
    loss.backward()
    
    #update
    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(k, loss.data)


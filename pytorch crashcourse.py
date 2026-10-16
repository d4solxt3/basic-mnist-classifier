import torch
import numpy as np
import math 
import torch.nn as nn

# 1. TENSORS

# a tensor is a multidimensional matrix containing elements of a single data type
# x = torch.empty(1)
x_2 = torch.empty(2, 3)
x_3 = torch.empty(2, 3, 5) # and so on and so forth

# you can also check the size of a tensor with the following command
# print("size", x.size()) # this returns x.size
# print("shape", x.shape()) # this returns the shape of x

# you can also check the data type
# print(x.dtype)
y = torch.zeros(5, 3, dtype=torch.float16) # you can also specify the data type (base is float 32)

# constructing a tensor with pre-existing data is possible with torch.tensor
z = torch.tensor([5.5, 3])
print (z, z.dtype)

# use the requires_grad function for all variables that you want to optimise in the model
z_2 = torch.tensor([5.5, 3], requires_grad=True) # by default, the require grad is false.
print(z_2)

# there are a few operations that can be done with tensors, variables a, b and c will be used to illustrate this:

a = torch.ones(2, 3)
b = torch.rand(2, 3)

# elementwise addition can be performed 
c = a + b
# torch.add(a, b)

# furthermore, anything with an inline underscore is considered as a operation:
# a.add_(b)

print(a)
print(b)
print(c)

# the same applies to the other arithemtic operations:

# subtraction
d = a - b
# torch.sub(a, b)

# multiplication
e = a * b
torch.mul(a, b)

# slicing
x = torch.rand(5, 6)
print(x)
print("x[:, 0]", x[:, 0]) # all rows, column 0
print("x[1, :]", x[1, :]) # all columns row 1
print("x[1,1]", x[1,1]) # element at (1, 1)

# you can also get the actual value if there is only one element in your tensor

print("x[1, 1].item()", x[1,1].item()) # this gives the actual float value

# you can reshape the tensor using torch.view or calling x.view
j = torch.rand(4, 4)
k = j.view(16)
l = j.view(-1, 8)

# if it is -1, pytorch will infer from the other dimensions the necessary size of the new tensor
# in this case it needs 2 to get to a size 16

# you can also convert a tensor to a numpy array
p = torch.rand(5, 4)
print(p)

# to do this, you can just call using .numpy

q = p.numpy()
print(q)
print(type(q))

# if the tensor is on the cpu, both of them will share the memory location, so changing one changes the other. 

# if you want to import from numpy to torch, use torch.from_numpy() and np.ones (basically just create an array in numpy and use that. 
v = np.ones(5)
w = torch.from_numpy(v)
v_2 = torch.tensor(v)
print(v)
print(w)
print(v_2)

# you have to be careful when modifying the tensors.
v += 1
print(v)
print(w)
print(v_2)

# by default, tensors are created on the cpu, but they can be created on the gpu or moved to the gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = x.to("cpu")
# x = x.to("cuda")

x = torch.rand(2, 2, device=device) # this is how you directly create them on a cpu.

# 2. AUTOGRAD -- automatic differentiation for tensors, computes partial derivatives while applying the chain rule for backprop.

# for this, we can use requires_grad=True

x_j = torch.rand(2, 3, requires_grad=True)
y_j = x_j + 2

# since y was created as part of a tensor operation, it has a grad_fn attribute

print(x_j) # this is created by the user, so grad_fn = none
print(y_j)
print(y_j.grad_fn)

# grad_fn for y_j will be backwards (it is the gradient function)

z_j = y_j * y_j * 2
print(z_j)
z_j = z_j.mean()
print(z_j)

# to get the gradient, you can just call z_j.backward()

z_j.backward()
print(x_j.grad)

# !!! calling .backward() accumulates the gradient for this tensor into .grad attribute
# use optimizer.zero_grad() to make sure that is emptied in each training iteration

a = torch.randn(2, 3)
b = (a * a).sum()
print(a.requires_grad)
print(b.grad_fn)

# you can change this even if it is an existing flag in place

a.requires_grad=True
b = (a * a).sum()
print(a.requires_grad)
print(b.grad_fn)

# .detach creates a new tensor with requires_grad=False, this can be applied to any tensors.

a = torch.rand(2, 2, requires_grad=True)
b = a.detach()
print(a.requires_grad)
print(b.requires_grad)

# the last way to do this is to wrap with torch.no_grad()

a = torch.rand(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    b = a ** 2
    print(b.requires_grad)

# 3. LINEAR REGRESSION WITH AUTOGRAD

    # LINEAR REGRESSION = f(x) = w * x + b where w is the weights and b the bias
    # In this case, take f(x) = 2 * x

X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16], dtype=torch.float32)

w = torch.tesnor(0.0, dtype=torch.float32, requires_grad=True) # initialise the tensor with 0 and set requires_grad=True

# model output
def forward(x):
    return w * x

# loss = mean squared error
def loss(y, y_pred):
    return((y_pred - y) ** 2).mean()

X_test = 5.0

print(f'Prediction before training: f({X_test}) = {forward(X_test)}.item:. 3f')


# TRAINING AN NEURAL NETWORK WITH REGRESSION
    
    # DEFINE A LEARNING RATE learning_rate = p
    # DEFINE NUMBER OF EPOCHS n_epoch = q

# TRAINING CODE

learning_rate = 0.01
n_epochs = 100

y_pred = forward(x) # determine the weights using the formula weights * x 

l = loss(y, y_pred)

# !!! CALCULATE THE WEIGHTS USING .backwards
l.backward()

with torch.no_grad():
    w -= learning_rate * w_grad

# you then 0 the gradient after learning for the process of gradient descent:
w.grad.zero_()

if (epoch+1) % 10 == 0:
    print(f'epoch {epoch + 1}: w = {w_item():. 3f}, loss = {l.item():. 3f}')

print(f'Prediction after learning: f({X_test}) = {forward(X_test).item():. 3f}')


# 4. MODEL LOSS AND OPTIMIZER

    # YOU DONT HAVE TO DEFINE YOUR LOSS FUNCTION -- PYTORCH HAS THESE BUILT IN
    
        # 1. DESIGN YOUR MODEL (INPUT, OUTPUT, FORWARD PASS ETC)
        # 2. CONSTRUCT YOUR LOOP AND OPTIMISER
        # 3. TRAINING LOOP
            # FORWARD = COMPUTE PREDICTION AND LOSS
            # BACKWWARD = COMPUTE THE GRAIDENTS
            # UPDATE THE WEIGHTS

# 0) Training Samples, the shape has to look like this:

X = torch.tensor([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8], [10], [12], [14], [16]], dtype=torch.float32)

n_samples, n_features = P.shape
print(f'n_samples = {n_samples}, n_features = {n_features}')

# 0) CREATE A TEST SAMPLE:

X_test = torch.tensor([[5], dtype=torch.float32])

# 1) Design model, it should implement the forward pass:

    # THE INIT FUFNCTION IS USED TO DEFINE THE DIFFERENT LAYERS
    # IN THE CASE OF LINEAR REGRESSION,  THIS IS JUST ONE LAYER, BUT THIS IS ALL THE LAYERS OF YOUR NEURAL NET.

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # You can then define the different layers 
        self.lin = nn.Linear(input.dim, output.dim)

        def forward(self, x):
            return self.lin(x)
    
input_size, output_size = n_features, n_features

model = LinearRegression(input_size, output_size)

print (f'Prediction before learning: f({X_test.item()}) = {model.(X_test).item():. 3f}')

# 2) Then define the loss and optimiser
learning_rate = 0.01
n_epochs = 100

loss = nn.MSEloss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Then design the training loop:

for epoch in range(n_epochs):
    # predict = forward pass with our model
    y_predicted = model(X)

    l = loss(Y, y_predicted)
    l.backward()
    optimizer.step()

    # AS ALWAYS, EMPTY THE GRADIENT BY CALLING ZERO GRAD

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        w, b = model.parameters()
        print('epoch', epoch+1, ' : w = ', w[0][0].item(), 'loss = ', l.item())

print (f'Prediction after learning: f({X_test.item()}) = {model.(X_test).item :. 3f}')


# FORWARD PASS --> LOSS --> optimizer.step and optimizer.zero_grad 



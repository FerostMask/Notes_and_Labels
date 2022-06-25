import torch

# ? Input 1
x = torch.arange(12).resize(3, 1, 4)
y = torch.randn(8).resize(2, 4)
z = x * y
# print(z)
print(x > y)

# # ? Input 2
# print(x.shape)

before = id(x)
x = x + y
print(before == id(x))
before = id(x)
x += y
print(before == id(x))

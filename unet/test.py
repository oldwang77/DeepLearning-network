import torch

A = torch.ones(2, 3)
B = 2*torch.ones(2, 2)

print("A = ", A)
print("B = ", B)

# C = torch.cat([A, B], dim=0)
# print("C = ", C)

D = torch.cat([A,B],dim=1)
print("D = ", D)

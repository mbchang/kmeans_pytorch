import einops as eo

import torch

def cosine(data1, data2=None):
    if data2 is None:
        data2 = data1

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis

def euclidean(data1, data2=None):
	r'''
	using broadcast mechanism to calculate pairwise ecludian distance of data
	the input data is N*M matrix, where M is the dimension
	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
	'''
	if data2 is None:
		data2 = data1 

	#N*1*M
	A = data1.unsqueeze(dim=1)

	#1*N*M
	B = data2.unsqueeze(dim=0)

	dis = (A-B)**2.0
	#return N*N matrix for pairwise distance
	dis = dis.sum(dim=-1).squeeze()
	return dis

def iou(data1, data2, eps=1e-7):
    """
        data1: (N, dim, num_classes)
        data2: (M, dim, num_classes)

        return a N*M matrix for pairwise distance
    """
    A = eo.rearrange(data1, 'n d k -> n 1 d k')
    B = eo.rearrange(data2, 'm d k -> 1 m d k')

    intersection = eo.reduce(A * B, 'n m d k -> n m k', 'sum')
    cardinality = eo.reduce(A + B, 'n m d k -> n m k', 'sum')
    union = cardinality - intersection
    jacc_loss = eo.reduce(intersection / (union + eps), 'n m d -> n m', 'sum')
    return - jacc_loss # (N, M)


def binary_iou(data1, data2, eps=1e-7):
    """
        data1: (N, dim)
        data2: (M, dim)

        return a N*M matrix for pairwise distance
    """
    data1 = torch.stack((data1, 1-data1), dim=-1)  # n d 2
    data2 = torch.stack((data2, 1-data2), dim=-1)  # m d 2

    A = eo.rearrange(data1, 'n d k -> n 1 d k')
    B = eo.rearrange(data2, 'm d k -> 1 m d k')

    intersection = eo.reduce(A * B, 'n m d k -> n m k', 'sum')
    cardinality = eo.reduce(A + B, 'n m d k -> n m k', 'sum')
    union = cardinality - intersection
    jacc_loss = eo.reduce(intersection / (union + eps), 'n m d -> n m', 'sum')
    return - jacc_loss # (N, M)
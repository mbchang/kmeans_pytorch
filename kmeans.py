import torch
import numpy as np

def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	initial_state = X[indices]
	return initial_state


# def lloyd(X, k, metric, tol=1e-4):

# 	initial_state = forgy(X, k)

# 	while True:
# 		dis = metric(X, initial_state)

# 		choice_cluster = torch.argmin(dis, dim=1)

# 		initial_state_pre = initial_state.clone()

# 		for index in range(k):
# 			selected = torch.nonzero(choice_cluster==index).squeeze()

# 			selected = torch.index_select(X, 0, selected)
# 			initial_state[index] = selected.mean(dim=0)
		

# 		center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

# 		if center_shift ** 2 < tol:
# 			break

# 	return choice_cluster, initial_state


def lloyd(X, k, metric, tol=1e-4):

	initial_state = forgy(X, k)

	i = 0
	while True:
		dis = metric(X, initial_state)

		choice_cluster = torch.argmin(dis, dim=1)

		initial_state_pre = initial_state.clone()

		for index in range(k):
			selected = torch.nonzero(choice_cluster==index).squeeze()

			selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
			if selected.shape[0] == 0:
				selected = X[torch.randint(len(X), (1,))]

			initial_state[index] = selected.mean(dim=0)
		
		center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
		print(f'iteration {i}\tcenter shift {center_shift}\ttolerance {tol}')
		i += 1

		if center_shift ** 2 < tol:
			break

	return choice_cluster, initial_state
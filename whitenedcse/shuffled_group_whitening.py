import torch
import torch.nn as nn
import math

# try:
#     torch.randn(2,2).to('cuda').svd()
#     linalg_device = 'cuda'
# except Exception:
linalg_device = 'cpu'

class ShuffledGroupWhitening(nn.Module):
    def __init__(self, num_features, num_groups=128, num_pos = 3, shuffle=True, engine='symeig'):
        super(ShuffledGroupWhitening, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        if self.num_groups is not None:
            assert self.num_features % self.num_groups == 0
        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_covariance', torch.eye(self.num_features))
        self.shuffle = shuffle if self.num_groups != 1 else False
        self.engine = engine
        self.num_pos = num_pos 

    def forward(self, x):
        N, D = x.shape
        if self.training:
            
            batch = int(N/3)
            x1 = x[:batch]
            x2 = x[batch: 2*batch]
            x3 = x[2*batch: ]
            if self.num_groups is None:
                G = math.ceil(2*D/N) # automatic, the grouped dimension 'D/G' should be half of the batch size N
                # print(G, D, N)
            else:
                G = self.num_groups
            if self.shuffle:
                new_idx1 = torch.randperm(D)
                x1 = x1.t()[new_idx1].t()
                new_idx2 = torch.randperm(D)
                x2 = x2.t()[new_idx2].t()
                new_idx3 = torch.randperm(D)
                x3 = x3.t()[new_idx3].t()
            
            x1 = x1.view(batch, G, D//G)
            x2 = x2.view(batch, G, D//G)
            x3 = x3.view(batch, G, D//G)
            x1 = (x1 - x1.mean(dim=0, keepdim=True)).transpose(0,1) # G, N, D//G
            x2 = (x2 - x2.mean(dim=0, keepdim=True)).transpose(0,1) # G, N, D//G
            x3 = (x3 - x3.mean(dim=0, keepdim=True)).transpose(0,1) # G, N, D//G
            # covs = x.transpose(1,2).bmm(x) / (N-1) #  G, D//G, N @ G, N, D//G -> G, D//G, D//G
            covs1 = x1.transpose(1,2).bmm(x1) / batch
            covs2 = x2.transpose(1,2).bmm(x2) / batch
            covs3 = x3.transpose(1,2).bmm(x3) / batch
            W1 = transformation(covs1, x1.device, engine=self.engine)
            W2 = transformation(covs2, x2.device, engine=self.engine)
            W3 = transformation(covs3, x3.device, engine=self.engine)
            
            x1 = x1.bmm(W1)
            x2 = x2.bmm(W2)
            x3 = x3.bmm(W3)
            if self.shuffle:
                x1 = x1.transpose(1,2).flatten(0,1)[torch.argsort(new_idx1)].t()
                x2 = x2.transpose(1,2).flatten(0,1)[torch.argsort(new_idx2)].t()
                x3 = x3.transpose(1,2).flatten(0,1)[torch.argsort(new_idx3)].t()
                x = torch.concat((x1, x2), 0)
                x = torch.concat((x, x3), 0)
                # return x.transpose(1,2).flatten(0,1)[torch.argsort(new_idx)].t()
                return x 

            else:
                x1 = x1.transpose(0,1).flatten(1)
                x2 = x2.transpose(0,1).flatten(1)
                x3 = x3.transpose(0,1).flatten(1)
                x = torch.concat((x1, x2), 0)
                x = torch.concat((x, x3), 0)
                return x 
                # return x.transpose(0,1).flatten(1)
        else:
            if self.num_groups is None:
                G = math.ceil(2*D/N) # automatic, the grouped dimension 'D/G' should be half of the batch size N
                # print(G, D, N)
            else:
                G = self.num_groups

            batch = N/3
            if self.shuffle:
                new_idx = torch.randperm(D)

            
            if self.shuffle:
                new_idx = torch.randperm(D)
                x = x.t()[new_idx].t()
            x = x.view(N, G, D//G)
            x = (x - x.mean(dim=0, keepdim=True)).transpose(0,1) # G, N, D//G
            # covs = x.transpose(1,2).bmm(x) / (N-1) #  G, D//G, N @ G, N, D//G -> G, D//G, D//G
            covs = x.transpose(1,2).bmm(x) / N
            # print(covs.shape)
            W = transformation(covs, x.device, engine=self.engine)
            x = x.bmm(W)
            if self.shuffle:
                return x.transpose(1,2).flatten(0,1)[torch.argsort(new_idx)].t()
            else:
                return x.transpose(0,1).flatten(1)


def transformation(covs, device, engine='symeig'):
    covs = covs.to(linalg_device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(linalg_device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1,2).to(x.device)
    else:
        covs = torch.tensor(covs, dtype=torch.float32, requires_grad=True)
        if engine == 'symeig':
            S, U = torch.symeig(covs.to(linalg_device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(linalg_device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(linalg_device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(linalg_device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1,2))
    return W


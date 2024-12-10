import torch.nn as nn
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(
        self, 
        CPRank = 64,
        latent_size=256, 
        dims=[512, 512, 512, 512, 512, 512, 512, 512], 
        dropout=[0, 1, 2, 3, 4, 5, 6, 7], 
        dropout_prob=0.2,
        norm_layers=[0, 1, 2, 3, 4, 5, 6, 7], 
        latent_in=[4], 
        weight_norm=False,
        xyz_in_all=False, 
        use_tanh=False, 
        latent_dropout=False):
        super(Decoder, self).__init__()

        self.CPRank = CPRank
        self.dtype = torch.cuda.FloatTensor
        self.CPdiag = nn.Parameter(torch.ones(CPRank).type(self.dtype))
        self.latentVec = nn.Embedding(1, latent_size).weight
        dims = [latent_size + 3] + dims + [CPRank*3] #[256+3, 512*8, 1] 一共10层

        self.num_layers = len(dims) #10
        self.norm_layers = norm_layers
        self.latent_in = latent_in #[4]
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(self, "linear" + str(layer), nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)))
            else:
                setattr(self, "linear" + str(layer), nn.Linear(dims[layer], out_dim))

            if ((not weight_norm) and self.norm_layers is not None and layer in self.norm_layers):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, xyz, mode='pointWise'):
        K = xyz.shape[0]
        latent_vecs = self.latentVec.expand(K, -1)
        latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
        x = torch.cat([latent_vecs, xyz], 1)

        for layer in range(0, self.num_layers - 1):
            linear = getattr(self, "linear" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, latent_vecs, xyz], dim=1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], dim=1)
            x = linear(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (self.norm_layers is not None and layer in self.norm_layers and not self.weight_norm):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)
        
        U, V, W = torch.chunk(x, 3, dim=1)
        output, nuNormLoss = self.tuckerForward(U, V, W, mode=mode)

        return output, nuNormLoss
    
    def tuckerForward(self, U, V, W, centre=None, mode='pointWise'):
        if centre is None:
            centre = torch.zeros((self.CPRank, self.CPRank, self.CPRank)).type(self.dtype)
            for i in range(self.CPRank):
                centre[i, i, i] = self.CPdiag[i]      
        if mode == 'Tucker':
            output = torch.einsum('abc,ia,jb,kc -> ijk', centre, U, V, W)
        elif mode == 'pointWise': #把n提出来，就变成了三个向量和core的组合，空间维度为1*1*1，计算n次
            output = torch.einsum('abc,na,nb,nc -> n', centre, U, V, W)
        else:
            raise NotImplementedError

        nuNormLoss = torch.norm(self.nuclear_norm(U), 1) +\
                     torch.norm(self.nuclear_norm(V), 1) +\
                     torch.norm(self.nuclear_norm(W), 1)

        return output, nuNormLoss
    
    def nuclear_norm(self, matrix):
        return torch.sum(torch.svd(matrix, compute_uv=False).S)
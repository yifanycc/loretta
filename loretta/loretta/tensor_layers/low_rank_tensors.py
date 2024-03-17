import torch
import math



class TensorTrain(torch.nn.Module):
    def __init__(self,config):

        super(TensorTrain, self).__init__()

        self.config = config
        self.factors = torch.nn.ParameterList()
        self.rank_parameters = []
        self.order = len(config.shape)
        self.build_factors_Gaussian()

    def build_factors_uniform(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = len(shape)
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)

        for i in range(1,order):
            x = torch.ones(ranks[i])
            self.rank_parameters.append(x)

    def build_factors_Gaussian(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = self.order
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)

        for i in range(1,order):
            x = torch.ones(ranks[i])
            self.rank_parameters.append(x)
    
    def update_factors(self):
        for U,D in zip(self.factors[:-1],self.rank_parameters):
            y = torch.nn.functional.threshold(D,0,0)
            U.data = U.data*y.data[None,None,:]
            D.data = torch.ones(D.shape,device=D.device)
    
    def get_rank_mask(self,threshold=1e-2):
        mask = []
        for i,x in enumerate(self.rank_parameters):
            x = torch.nn.functional.threshold(x,threshold,-100)
            self.rank_parameters[i].data = x
            y = torch.nn.functional.threshold(x,0,0)
            mask.append(y)
        return mask
    
    def estimate_rank(self,threshold=1e-2):
        out = []
        tol = threshold
        for x in self.rank_parameters:
            out.append(int(torch.sum(x>tol)))
        return out

    def get_factors(self,prune_mask=False,threshold = 1e-2):
        if prune_mask==False:
            return self.factors
        else:
            factors = []
            mask = self.get_rank_mask(threshold=threshold)
            for U,D in zip(self.factors[:-1],mask):
                # y = torch.nn.functional.threshold(D,0,0)
                D = D[None,None,:]
                factors.append(U*D)
            factors.append(self.factors[-1])
            return factors
    
    def get_full(self,factors):
        with torch.no_grad():
            out = factors[0]
            for U in factors[1:]:
                out = torch.tensordot(out,U,[[-1],[0]])
            
        return torch.squeeze(out)




class TensorTrainMatrix(torch.nn.Module):
    def __init__(self,config):

        super(TensorTrainMatrix, self).__init__()

        self.config = config
        self.factors = torch.nn.ParameterList()
        self.rank_parameters = torch.nn.ParameterList()
        self.order = len(config.shape[0])
        self.build_factors_Gaussian()

    def build_factors_uniform(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = len(shape)
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n = shape[i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)

        for i in range(1,order):
            x = torch.nn.Parameter(torch.ones(ranks[i]))
            self.rank_parameters.append(x)

    def build_factors_Gaussian(self):
        config = self.config
        shape = config.shape
        ranks = config.ranks
        order = self.order
        if type(ranks)==int:
            ranks = [1]+[ranks]*(order-1)+[1]
        
        for i in range(order):
            n1 = shape[0][i]
            n2 = shape[1][i]
            r1 = ranks[i]
            r2 = ranks[i+1]
            U = torch.nn.Parameter(torch.randn(r1,n1,n2,r2)/math.sqrt(r2)*self.config.target_sdv**(1/order))
            self.factors.append(U)

        for i in range(1,order):
            x = torch.nn.Parameter(torch.ones(ranks[i]))
            self.rank_parameters.append(x)
    
    def update_factors(self):
        for U,D in zip(self.factors[:-1],self.rank_parameters):
            y = torch.nn.functional.threshold(D,0,0)
            U.data = U.data*y.data[None,None,None,:]
            D.data = torch.ones(D.shape,device=D.device)
    
    def get_rank_mask(self,threshold=1e-2):
        mask = []
        for i,x in enumerate(self.rank_parameters):
            x = torch.nn.functional.threshold(x,threshold,-100)
            self.rank_parameters[i].data = x
            y = torch.nn.functional.threshold(x,0,0)
            mask.append(y)
        return mask
    
    def estimate_rank(self,threshold=1e-2):
        out = []
        tol = threshold
        for x in self.rank_parameters:
            out.append(int(torch.sum(x>tol)))
        return out

    def get_factors(self,prune_mask=False,threshold = 1e-2):
        if prune_mask==False:
            return self.factors
        else:
            factors = []
            mask = self.get_rank_mask(threshold=threshold)
            for U,D in zip(self.factors[:-1],mask):
                y = torch.nn.functional.threshold(D,0,0)
                D = D[None,None,None,:]
                factors.append(U*D)
            factors.append(self.factors[-1])
            return factors
    
    def get_full(self,factors):
        with torch.no_grad():
            out = factors[0]
            for U in factors[1:]:
                out = torch.tensordot(out,U,[[-1],[0]])
            
        return torch.squeeze(out)





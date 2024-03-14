import torch
import numpy as np
import tensorly as tl
from functools import reduce
# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize

def get_tensorized_index(idx,cum_prod):
    rem = idx
    out = []
    for x in cum_prod:
        val,rem = torch_divmod(rem,x) 
        out.append(val)

    out.append(rem)
    out = torch.stack(out).T
    return out.to(idx.device)

def torch_divmod(x,y):
    
    return x//y,torch.fmod(x,y)

def get_cum_prod(shape):
    cum_prod = [1]
    for x in reversed(shape[0][1:]):
        cum_prod.append(x*cum_prod[-1])

    cum_prod.reverse()
    cum_prod.pop()
    return cum_prod

def tensorized_lookup(idx,factors,cum_prod,shape,tensor_type):

    tensorized_indices = get_tensorized_index(idx,cum_prod) 

    if tensor_type == 'TensorTrainMatrix':
        gathered_rows = ttm_gather_rows(factors,tensorized_indices,shape)
    elif tensor_type == 'TensorTrain':
        gathered_rows = tt_gather_rows(factors,tensorized_indices,shape)
    elif tensor_type =='CP':
        gathered_rows = cp_gather_rows(factors,tensorized_indices,shape)
    elif tensor_type == 'Tucker':
        gathered_rows = tucker_gather_rows(factors,tensorized_indices,shape)

    return gathered_rows

def tucker_gather_rows(factors,tensorized_indices,shape):
    full_factors = factors[1]
    core = factors[0]

    tmp_factors = []
    for i,col in enumerate(tensorized_indices.unbind(1)):
        tmp_factors.append(full_factors[i][col,:])

    tmp_core = core

    tmp_core = tl.tenalg.mode_dot(tmp_core,tmp_factors[0],0)

    tmp_core = tmp_core.T

    for factor in tmp_factors[1:]:
        tmp_core = tmp_core*factor.T
        tmp_core = tmp_core.sum(-2)

    tmp_core = tmp_core.T
    #tmp_core.shape

    for i,factor in enumerate(factors[1][len(shape[0]):]):

        tmp_core = tl.tenalg.mode_dot(tmp_core,factor,i-len(shape[1]))

    gathered_rows = tmp_core.reshape(-1,np.prod(shape[1]))
    return gathered_rows

def cp_gather_rows(factors,tensorized_indices,shape):

    full_factors = factors

    tmp_factors = []

    for i,col in enumerate(tensorized_indices.unbind(1)):
        tmp_factors.append(full_factors[i][col,:])

    reduced = reduce(lambda x,y:x*y,tmp_factors)

    tmp_factors = [reduced]

    for factor in full_factors[-len(shape[1]):]:
        tmp_factors.append(factor)

    gathered_rows = tl.kruskal_to_tensor((None,tmp_factors)).view(-1,np.prod(shape[1]))

    return gathered_rows

def tt_reduce_fun(x,y):
    return torch.bmm(x,y.permute([1,0,2]))
#elif tensor_type =='TensorTrain':

def tt_gather_rows(cores,tensorized_indices,shape):

    tmp_cores = []
    for i,col in enumerate(tensorized_indices.unbind(1)):
        tmp_cores.append(cores[i][:,col,:])

    tmp_cores[0] = tmp_cores[0].permute([1,0,2])

    reduced = reduce(tt_reduce_fun,tmp_cores)








    tmp_factors = [reduced.permute([1,0,2])]

    for core in cores[-len(shape[1]):]:
        tmp_factors.append(core)


    batch_tensor = tl.tt_to_tensor(tmp_factors).view(-1,np.prod(shape[1]))

    return batch_tensor


def get_ttm_cum_prod(dims_0):

    out = []
    rem = idx

    cum_prod = [1]
    for x in reversed(dims_0[1:]):
        cum_prod.append(x*cum_prod[-1])

    cum_prod.reverse()
    cum_prod.pop()

    return cum_prod

def ttm_gather_rows(cores, inds,shape):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """

    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(core.shape[0]) for core in cores] + [1, ]

    for k, core in enumerate(cores):
        i = inds[:, k]

        # print(torch.unique(i).shape)
 
        cur_slice = torch.index_select(core, 1, i)
        # cur_slice = core[:,i,...]
        # r x B x M x r

        # print(core.shape)

        if k == 0:
            res = cur_slice.transpose(0, 1)
            # B x r x M x r

        else:
            res = res.contiguous().view(batch_size, -1, ranks[k])
            # B x rM x r
            curr_core = cur_slice.view(ranks[k], batch_size, -1)
            # r x B x Mr
            res = torch.einsum('oqb,bow->oqw', (res, curr_core))
    res = torch.einsum('i...i->...', res.view(batch_size, ranks[0], res.shape[1] // ranks[0], -1, ranks[0]).transpose(0, 1))
    # print(res.shape)
    return res.reshape(-1,np.prod(shape[1]))



def TTM_lookup_LP(idx,cores,cum_prod,shape,rounding,bits):
    """
    inds -- list of indices of shape batch_size x d
    d = len(tt_mat.raw_shape[1])
    """
    inds = get_tensorized_index(idx,cum_prod) 
    shape = shape

    slices = []
    batch_size = int(inds.shape[0])

    ranks = [int(core.shape[0]) for core in cores] + [1, ]

    factors = []
    for k, core in enumerate(cores):
        i = inds[:, k]

        # print(torch.unique(i).shape)
 
        cur_slice = torch.index_select(core, 1, i)
        
        factors.append(cur_slice)

    

    res = TTM_emb.apply(rounding,bits,*factors)
    # print(res.shape)
    return res.reshape(-1,np.prod(shape[1]))



class TTM_emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rounding, bits, *factors):
        ctx.bits = bits
        ctx.rounding = rounding
        ctx.factors = factors
        ctx.left = []
        
        # Q = lambda x: float_quantize(x, exp=bits[1], man=bits[2], rounding=rounding)
        
        # out = torch.squeeze(factors[0])
        out = factors[0]
        ctx.left.append(out)
        for U in factors[1:]:
            out = torch.einsum('rbmp,pbnq->rbmnq',out,U)
            out = (torch.flatten(out,start_dim=2,end_dim=3))
            ctx.left.append(out)
            
        
        # out = torch.squeeze(out)
        return out 

    @staticmethod
    def backward(ctx, dy):
        # Q = lambda x: float_quantize(x, exp=ctx.bits[1], man=ctx.bits[2], rounding=ctx.rounding)
        
        factors = ctx.factors
        
        m = [U.shape[2] for U in factors]
        b = dy.shape[1]
        
        scale_y = max((torch.mean(torch.abs(dy))+0*torch.sqrt(torch.var(torch.abs(dy))))*(1e-2),1e-8)
        
        
        # Q = lambda x:x
        # scale_y = 1
        
        # dy_Q = float_quantize(dy/scale_y, exp=ctx.bits[1], man=ctx.bits[2], rounding=ctx.rounding)
        
        # print('scale',scale_y)
        
        # print(torch.norm(dy-dy_Q)/torch.norm(dy))
        # print(torch.min(torch.abs(dy_Q)))
        # print(torch.max(torch.abs(dy_Q)))
            
        # dy = dy_Q
        
        
        
        # print(torch.mean(torch.abs(dy)))
        
        # dy = torch.reshape(dy,[b]+m)
        
        left = ctx.left
        
        
        grads = []
        right_U = dy
        

        
        for i in range(len(factors)-1):
            j = -i-1
            left_U = left[j-1]
            

            right_U = torch.reshape(right_U,[right_U.shape[0],b,-1,m[j],right_U.shape[-1]])

            
            grad = torch.einsum('rbmp,sbmnq->rpbnsq',left_U,right_U)
            grads = [grad.reshape(grad.shape[1:-1])*scale_y] + grads
            
            right_U = torch.einsum('rbmp,pbnmq->rbnq',factors[j],right_U)
            
        grads = [right_U.transpose(0,-1)*scale_y]+grads
        



        
        return None,None,tuple(grads)
        
        
        
            
        
        
        
        
        
        
        
        


"""
def convert_to_tt(idx,dims):
    out = []
    rem = idx

    for x in dims:
        val,rem = divmod(rem,int(x)) 
        out.append(val)
    return out
"""

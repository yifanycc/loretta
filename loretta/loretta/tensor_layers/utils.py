import torch
import numpy as np
# from qtorch.quant import fixed_point_quantize, block_quantize, float_quantize
import torch.nn.functional as F

class config_class():
    def __init__(self,
                **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))




class quantize(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    pass


class TT_forward_quant(torch.autograd.Function):
    pass



class TT_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, *factors):

        with torch.no_grad():

            tt_shape = [U.shape[1] for U in factors]
            ndims = len(factors)
            d = int(ndims / 2)

            ctx.input_shape = matrix.shape
            if len(matrix.shape)==3:
                out_shape = [matrix.shape[0],matrix.shape[1],np.prod(list(tt_shape[d:]))]
                matrix = torch.flatten(matrix,start_dim=0,end_dim=1)
            else:
                out_shape = [matrix.shape[0],np.prod(list(tt_shape[d:]))]
            ctx.out_shape = out_shape

            ctx.factors = factors
            ctx.matrix = matrix


            
    
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            matrix_cols = matrix.shape[0]

            saved_tensors = [matrix]
            left = []
            right = []

            output = factors[0].reshape(-1, ranks[1])
            left.append(output)

            for core in factors[1:d]:
                output = (torch.tensordot(output, core, dims=([-1], [0])))
                left.append(output)

            output = F.linear(matrix, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0))


            saved_tensors.append(left)

            temp = factors[d]
            right.append(temp)
            for core in factors[d + 1:]:
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))
                right.append(temp)


            
            output = F.linear(output, torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
                                            0, -1)).reshape(matrix_cols, np.prod(tt_shape_col)).reshape(*out_shape)
        
            
            saved_tensors.append(right)
            ctx.saved_tensors_custom = saved_tensors
       
   
        return output

       
    @staticmethod
    def backward(ctx, dy):
        with torch.no_grad():
            factors = ctx.factors
            ndims = len(factors)
            d = int(ndims / 2)
            ranks = [U.shape[0] for U in factors] + [1]
            tt_shape = [U.shape[1] for U in factors]
            tt_shape_row = list(tt_shape[:d])
            tt_shape_col = list(tt_shape[d:])
            saved_tensors = ctx.saved_tensors_custom

            
            
            if len(dy.shape)==3:
                dy = torch.flatten(dy,start_dim=0,end_dim=1)


            matrix = saved_tensors[0]
            left = saved_tensors[1]
            right = saved_tensors[2]
            left_grads = []
            right_grads = []

            dy_core_prod = right[-1]


        
            dy_core_prod = (torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1])))


            matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([0], [0]))


            for i in reversed(range(1, d)):
                grad = (torch.tensordot(left[i - 1].reshape(-1, ranks[i]),
                                    matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
                                                                ranks[d]),
                                    dims=([0], [0])))
                # print(grad.shape)
                if i == d - 1:
                    right_core = factors[i]
                else:
                    grad = (torch.tensordot(grad, right_core, dims=([2, 3], [1, 2])))

                    right_core = torch.tensordot(factors[i], right_core,
                                                dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
                
                if grad.shape != factors[i].shape:
                    grad = grad.reshape(list(factors[i].shape))
                # print(grad.shape)
                left_grads.append(grad)
            temp = (torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
                                            right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))


            left_grads.append(temp)

            left_grads = left_grads[::-1]

            matrix_core_prod = left[-1]
            matrix_core_prod = (torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
                                            matrix, dims=([0], [1])))

            
            # print('dx=',torch.max(matrix_core_prod))
            matrix_dy_core_prod = (torch.tensordot(matrix_core_prod, dy, dims=([1], [0])))


            for i in reversed(range(1, d)):
                grad = (torch.tensordot(right[i - 1].reshape(-1, ranks[d + i]),
                                    matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),
                                    dims=([0], [0])))
            
                if i == d - 1:
                    right_core = factors[d + i].reshape(-1, tt_shape_col[i])
                else:
                
                    grad = (torch.tensordot(grad, right_core, dims=([-1], [1])))
                    


                    right_core = (torch.tensordot(factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],-1))
                                                                                                                                                                            
                if grad.shape != factors[d + i].shape:
                    grad = grad.reshape(list(factors[i].shape))

                right_grads.append(grad)

            temp = (torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
                                            right_core, dims=([-1], [1])))

            right_grads.append(temp)

            right_grads = right_grads[::-1]

            dx = factors[-1].reshape(ranks[-2], -1)
            for core in reversed(factors[d:-1]):
                dx = (torch.tensordot(core, dx, dims=([-1], [0])))

        
            dx = (torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1])))



            temp = factors[0].reshape(-1, ranks[1])
            for core in factors[1:d]:
                temp = (torch.tensordot(temp, core, dims=([-1], [0])))


            dx = (torch.tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1])))
            dx = torch.reshape(dx,ctx.input_shape)            

            all_grads = [g for g in left_grads+right_grads]



        return dx, tuple(all_grads)
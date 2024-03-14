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
    #
    # @staticmethod
    # def forward(ctx, input, scale, bit=[1,7,0],rep='INT',rounding='nearest'):
    #     """
    #     In the forward pass we receive a Tensor containing the input and return
    #     a Tensor containing the output. ctx is a context object that can be used
    #     to stash information for backward computation. You can cache arbitrary
    #     objects for use in the backward pass using the ctx.save_for_backward method.
    #     """
    #
    #
    #     if rep=='INT':
    #         max_q = 2**(bit[1])-1
    #         min_q = -2**(bit[1])
    #
    #         quant = lambda x: fixed_point_quantize(x,wl=bit[1],fl=bit[2],rounding=rounding)
    #     elif rep=='FLOAT':
    #         exp = bit[1]
    #         man = bit[2]
    #         if exp==5:
    #             max_q = 5000
    #             min_q = -5000
    #         elif exp==4:
    #             max_q = 500
    #             min_q = -500
    #         else:
    #             max_q = 1e7
    #             min_q = -1e7
    #         quant = lambda x: float_quantize(x, exp=exp, man=man, rounding=rounding)
    #
    #
    #     ctx.save_for_backward(input, scale)
    #     ctx.quant = quant
    #     ctx.input_div_scale = input/scale
    #     ctx.q_input = quant(ctx.input_div_scale)
    #     ctx.min_q = torch.tensor(min_q)
    #     ctx.max_q = torch.tensor(max_q)
    #
    #     return scale * ctx.q_input
    #
    # @staticmethod
    # def backward(ctx, grad_output):
    #     """
    #     In the backward pass we receive a Tensor containing the gradient of the loss
    #     with respect to the output, and we need to compute the gradient of the loss
    #     with respect to the input.
    #     """
    #     input, scale= ctx.saved_tensors
    #     grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
    #
    #     grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))
    #
    #
    #     grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q.to(grad_scale.device), max = ctx.max_q.to(grad_scale.device))
    #
    #
    #     return grad_input, grad_scale, None, None, None
    #


class TT_forward_quant(torch.autograd.Function):
    pass
    # @staticmethod
    # def forward(ctx, rounding, bits,scale_med,scale_grad,scale_y, matrix, *factors):
    #
    #     Q = lambda x: float_quantize(x, exp=bits[1], man=bits[2], rounding=rounding)
    #
    #
    #     tt_shape = [U.shape[1] for U in factors]
    #     ndims = len(factors)
    #     d = int(ndims / 2)
    #
    #     ctx.input_shape = matrix.shape
    #     if len(matrix.shape)==3:
    #         out_shape = [matrix.shape[0],matrix.shape[1],np.prod(list(tt_shape[d:]))]
    #         matrix = torch.flatten(matrix,start_dim=0,end_dim=1)
    #     else:
    #         out_shape = [matrix.shape[0],np.prod(list(tt_shape[d:]))]
    #     ctx.out_shape = out_shape
    #
    #     ctx.bits = bits
    #     ctx.rounding = rounding
    #     ctx.factors = factors
    #     ctx.matrix = matrix
    #
    #     grad_scale = 0
    #     grad_scale_num = 1
    #
    #
    #     ndims = len(factors)
    #     d = int(ndims / 2)
    #     ranks = [U.shape[0] for U in factors] + [1]
    #     tt_shape = [U.shape[1] for U in factors]
    #     tt_shape_row = list(tt_shape[:d])
    #     tt_shape_col = list(tt_shape[d:])
    #     matrix_cols = matrix.shape[0]
    #
    #     saved_tensors = [matrix]
    #     left = []
    #     right = []
    #
    #     output = factors[0].reshape(-1, ranks[1])
    #     # print(torch.max(output))
    #     left.append(output)
    #     for core in factors[1:d]:
    #         # print(torch.max(core))
    #         output_ = torch.tensordot(output, core, dims=([-1], [0]))
    #         # print(torch.max(output_))
    #
    #
    #         output = Q(torch.tensordot(output, core, dims=([-1], [0])))
    #
    #         # print(torch.max(output))
    #
    #
    #         left.append(output)
    #
    #     # output,g = Q(F.linear(matrix, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0)))
    #     output = Q((matrix@torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0).T))
    #
    #
    #     saved_tensors.append(left)
    #
    #     temp = factors[d]
    #     right.append(temp)
    #     for core in factors[d + 1:]:
    #         temp = Q(torch.tensordot(temp, core, dims=([-1], [0])))
    #         right.append(temp)
    #
    #
    #     output = (output@torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
    #                                             0, -1).T).reshape(*out_shape)
    #
    #     saved_tensors.append(right)
    #
    #     ctx.saved_tensors_custom = saved_tensors
    #     ctx.grad_med = grad_scale
    #     ctx.scale_grad = scale_grad
    #     ctx.scale_y = scale_y
    #
    #
    #     return output
    #
    # @staticmethod
    # def backward(ctx, dy):
    #     factors = ctx.factors
    #     ndims = len(factors)
    #     d = int(ndims / 2)
    #     ranks = [U.shape[0] for U in factors] + [1]
    #     tt_shape = [U.shape[1] for U in factors]
    #     tt_shape_row = list(tt_shape[:d])
    #     tt_shape_col = list(tt_shape[d:])
    #     saved_tensors = ctx.saved_tensors_custom
    #
    #
    #
    #     Q = lambda x: float_quantize(x, exp=ctx.bits[1], man=ctx.bits[2], rounding=ctx.rounding)
    #
    #     # ctx.scale_y.data[0] = (torch.mean(torch.abs(dy))+0*torch.sqrt(torch.var(torch.abs(dy))))*(1e-2)
    #
    #
    #
    #
    #
    #     scale_y = max((torch.mean(torch.abs(dy))+0*torch.sqrt(torch.var(torch.abs(dy))))*(1e-2),1e-8)
    #
    #     dy = float_quantize(dy/scale_y, exp=ctx.bits[1], man=ctx.bits[2], rounding=ctx.rounding)
    #
    #     if len(dy.shape)==3:
    #         dy = torch.flatten(dy,start_dim=0,end_dim=1)
    #
    #
    #     matrix = saved_tensors[0]
    #     left = saved_tensors[1]
    #     right = saved_tensors[2]
    #     left_grads = []
    #     right_grads = []
    #
    #
    #     dy_core_prod = right[-1]
    #
    #
    #
    #     dy_core_prod = Q(torch.tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1])))
    #
    #
    #     matrix_dy_core_prod = torch.tensordot(matrix, dy_core_prod, dims=([0], [0]))
    #
    #     for i in reversed(range(1, d)):
    #
    #         grad = Q(torch.tensordot(left[i - 1].reshape(-1, ranks[i]),
    #                             matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
    #                                                         ranks[d]),
    #                             dims=([0], [0])))
    #
    #         if i == d - 1:
    #             right_core = factors[i]
    #         else:
    #             grad = Q(torch.tensordot(grad, right_core, dims=([2, 3], [1, 2])))
    #
    #             right_core = torch.tensordot(factors[i], right_core,
    #                                         dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
    #
    #         if grad.shape != factors[i].shape:
    #             grad = grad.reshape(list(factors[i].shape))
    #
    #         left_grads.append(grad)
    #     temp = Q(torch.tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
    #                                     right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))
    #
    #
    #     left_grads.append(temp)
    #
    #     left_grads = left_grads[::-1]
    #
    #     matrix_core_prod = left[-1]
    #     matrix_core_prod = Q(torch.tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
    #                                     matrix, dims=([0], [1])))
    #
    #
    #
    #     matrix_dy_core_prod = Q(torch.tensordot(matrix_core_prod, dy, dims=([1], [0])))
    #
    #
    #
    #     for i in reversed(range(1, d)):
    #         grad = Q(torch.tensordot(right[i - 1].reshape(-1, ranks[d + i]),
    #                             matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),
    #                             dims=([0], [0])))
    #
    #         if i == d - 1:
    #             right_core = factors[d + i].reshape(-1, tt_shape_col[i])
    #         else:
    #             grad = Q(torch.tensordot(grad, right_core, dims=([-1], [1])))
    #
    #
    #
    #             right_core = Q(torch.tensordot(factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],-1))
    #
    #         if grad.shape != factors[d + i].shape:
    #             grad = grad.reshape(list(factors[i].shape))
    #
    #         right_grads.append(grad)
    #
    #     temp = Q(torch.tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
    #                                     right_core, dims=([-1], [1])))
    #     right_grads.append(temp)
    #
    #     right_grads = right_grads[::-1]
    #
    #     dx = factors[-1].reshape(ranks[-2], -1)
    #     for core in reversed(factors[d:-1]):
    #         dx = Q(torch.tensordot(core, dx, dims=([-1], [0])))
    #
    #
    #     # print('dx=',torch.max(dx))
    #     dx = Q(torch.tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1])))
    #
    #
    #     temp = factors[0].reshape(-1, ranks[1])
    #     for core in factors[1:d]:
    #         temp = Q(torch.tensordot(temp, core, dims=([-1], [0])))
    #
    #
    #     dx = Q(torch.tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1])))
    #
    #     dx = torch.reshape(dx,ctx.input_shape)
    #
    #
    #
    #     dx = dx*scale_y
    #     all_grads = [g*scale_y for g in left_grads+right_grads]
    #
    #
    #
    #     return None,None,None,None,None,dx.to(dy.device), tuple(all_grads)
    #     # z = torch.tensor(0).to('cuda')
    #     # [print(U.shape) for U in left_grads+right_grads]
    #     # return None,z,z,z, dx, *(left_grads + right_grads)



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
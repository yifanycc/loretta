# -*- coding: utf-8 -*-
import numpy as np
from torch import nn as nn
import torch
import logging
import os
from torch.nn import functional as F
from torch import nn
import random
from tqdm import tqdm
seed = 1234
random.seed(seed)
np.random.seed(seed)

logger = logging.getLogger(__name__)

class config_class:
    def __init__(self, **kwargs):
        for x in kwargs:
            setattr(self, x, kwargs.get(x))


class Tensor:
    def __init__(self, tensor_input_shape, tensor_output_shape, truncate_num, fix_rank=None):
        self.tensor_input_shape = tensor_input_shape
        self.tensor_output_shape = tensor_output_shape
        self.truncate_num = truncate_num
        self.num_dim = len(tensor_input_shape)
        self.tensor_ranks = self.compute_rank(truncate_num=None)
        if fix_rank:
            self.tensor_truncate_ranks = fix_rank
        else:
            self.tensor_truncate_ranks = self.compute_rank(truncate_num=self.truncate_num)

    def compute_rank_position(self, s, truncate_num=None):

        """
        Calculate the rank position in Tensor bond dimension
        :param s: target bond ,type = int, range in [1:len(tensor_input_shape-1)], r_0 = r_n = 1.
        :return:  target bond 's' real bond dimension.
        """
        rank_left = 1  # ranks_left: all the shape multiply in left of 's'.
        rank_right = 1  # ranks_right: all the shape multiply in right of 's'.
        for i in range(0, s):
            rank_left = rank_left * self.tensor_input_shape[i] * self.tensor_output_shape[i]
        for i in range(s, self.num_dim):
            rank_right = rank_right * self.tensor_input_shape[i] * self.tensor_output_shape[i]
        if truncate_num == None:
            min_rank = min(rank_left, rank_right)
        else:
            min_rank = min(int(self.truncate_num), rank_left, rank_right)
        return min_rank

    def compute_rank(self, truncate_num):
        """
        :param tensor_input_shape: the input tensor shape, type = list. [i0,i1,i2,...,i_(n-1)]
        :param truncate_num: the truncate number of tensor, type = int.
        :return:max bond dimension in every bond position, type = list, [r0,r1,r2,...,r_n],r0=r_n=1
        """
        bond_dims = [1 for i in range(self.num_dim + 1)]
        for i in range(1, self.num_dim):
            bond_dims[i] = self.compute_rank_position(i, truncate_num)
        return bond_dims

    def get_tensor_set(self, inp_matrix):
        """
        Calculate the left canonical of input matrix with a given tensor_input_shape
        :param inp_matrix: the input matrix
        :param tensor_input_shape:
        :return: a tensor with left canonical in input matrix
        """
        tensor_set = []
        res = inp_matrix
        #################################################################################

        res = res.reshape(tuple(self.tensor_input_shape[:]) + tuple(self.tensor_output_shape[:]))
        self.index_permute = np.transpose(
            np.array(range(len(self.tensor_input_shape) + len(self.tensor_output_shape))).reshape((2, -1))).flatten()
        res = np.transpose(res, self.index_permute)
        #################################################################################
        for i in range(self.num_dim - 1):
            # Do the SVD operator
            res = res.reshape([self.tensor_ranks[i] * self.tensor_input_shape[i] * self.tensor_output_shape[i], -1])
            u, lamda, v = np.linalg.svd(res, full_matrices=False)
            # The first tensor should be T1(r_i+1, m_i, n_i, r_i)
            u = u.reshape([self.tensor_ranks[i], self.tensor_input_shape[i], self.tensor_output_shape[i], self.tensor_ranks[i + 1]])
            tensor_set.append(u)
            res = np.dot(np.diag(lamda), v)
        res = res.reshape([self.tensor_ranks[self.num_dim - 1], self.tensor_input_shape[self.num_dim - 1],
                           self.tensor_output_shape[self.num_dim - 1], self.tensor_ranks[self.num_dim]])
        tensor_set.append(res)
        return tensor_set

    def left_canonical(self, tensor_set):
        left_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        mat = tensor_set[0]
        mat = mat.reshape(-1, mat.shape[3])
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        left_canonical_tensor[1] = np.dot(np.diag(lamda), v)
        for i in range(1, self.num_dim - 1):
            mat = np.tensordot(left_canonical_tensor[i], tensor_set[i], [1, 0])
            mat = mat.reshape(-1, mat.shape[-1])
            u, lamda, v = np.linalg.svd(mat, full_matrices=False)
            left_canonical_tensor[i + 1] = np.dot(np.diag(lamda), v)
        return left_canonical_tensor

    def right_canonical(self, tensor_set):
        """
        Calculate the right tensor canonical for Tensor format required
        :param left_tensor: the tensor_set output from function: left_canonical
        :return: the right_tensor_canonical format for calculate the tensor decomposition
        """
        right_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        # print(tensor_set.shape)
        mat = tensor_set[self.num_dim - 1]
        mat = mat.reshape(mat.shape[0], -1)
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        right_canonical_tensor[self.num_dim - 1] = np.dot(u, np.diag(lamda))

        for i in range(self.num_dim - 2, 0, -1):
            mat = np.tensordot(tensor_set[i], right_canonical_tensor[i + 1], [3, 0])
            mat = mat.reshape(mat.shape[0], -1)
            u, lamda, v = np.linalg.svd(mat, full_matrices=False)
            right_canonical_tensor[i] = np.dot(u, np.diag(lamda))
        return right_canonical_tensor

    def expectrum_normalization(self, lamda):
        """
        Do the lamda normalization for calculate the needed rank for Tensor structure
        :param lamda: lamda parameter from left canonical
        :return:
        """
        norm_para = np.sum(lamda ** 2) ** (0.5)
        lamda_n = lamda / norm_para
        lamda_12 = lamda ** (-0.5)
        return lamda_n, np.diag(lamda_12)

    def gauge_aux_p_q(self, left_canonical_tensor, right_canonical_tensor):
        p = [0 for i in range(self.num_dim + 1)]
        q = [0 for i in range(self.num_dim + 1)]
        lamda_set = [0 for i in range(self.num_dim + 1)]
        lamda_set_value = [0 for i in range(self.num_dim + 1)]
        lamda_set[0] = np.ones([1, 1])
        lamda_set[-1] = np.ones([1, 1])
        for i in range(1, self.num_dim):
            mat = np.dot(left_canonical_tensor[i], right_canonical_tensor[i])
            # mat = right_canonical_tensor[i]
            u, lamda, v = np.linalg.svd(mat)
            lamda_n, lamda_l2 = self.expectrum_normalization(lamda)
            lamda_set[i] = lamda_n
            lamda_set_value[i] = lamda
            p[i] = np.dot(right_canonical_tensor[i], v.T)
            p[i] = np.dot(p[i], lamda_l2)
            q[i] = np.dot(lamda_l2, u.T)
            q[i] = np.dot(q[i], left_canonical_tensor[i])
        return p, q, lamda_set, lamda_set_value

    def tensor_canonical(self, tensor_set, p, q):
        tensor_set[0] = np.tensordot(tensor_set[0], p[1], [3, 0])
        tensor_set[-1] = np.tensordot(q[self.num_dim - 1], tensor_set[-1], [1, 0])
        for i in range(1, self.num_dim - 1):
            tensor_set[i] = np.tensordot(q[i], tensor_set[i], [1, 0])
            tensor_set[i] = np.tensordot(tensor_set[i], p[i + 1], [3, 0])
        return tensor_set

    def truncated_tensor(self, tensor_set, step_train=False):
        """
        Get a untruncated tensor by tensor
        :param tensor_set: the input weight
        :return: a untruncated tensor_set by tensor
        """
        if step_train:
            tensor_set_tmp = [i.detach().cpu().numpy() for i in tensor_set]
            cano_tensor_set = self.bi_canonical(tensor_set_tmp)
            tensor_set = torch.nn.ParameterList(
                [nn.Parameter(torch.from_numpy(i).cuda(), requires_grad=True) for i in cano_tensor_set])
            tensor_set[2].requires_grad = False

        tensor_trunc = self.tensor_truncate_ranks[:]
        for i in range(self.num_dim):
            if step_train:
                mask_noise = torch.ones_like(tensor_set[i])
            t = tensor_set[i]
            r_l = tensor_trunc[i]
            r_r = tensor_trunc[i + 1]
            if isinstance(tensor_set[i], nn.parameter.Parameter):
                if step_train:

                    mask_noise[r_l:, :, :, :] = 0.0
                    mask_noise[:r_l, :, :, r_r:] = 0.0
                    tensor_set[i].data = tensor_set[i].data * mask_noise
                else:
                    tensor_set[i].data = t[:r_l, :, :, :r_r]
            else:
                tensor_set[i] = t[:r_l, :, :, :r_r]
                assert "Check! tensor_set is not nn.parameter.Parameter"
        return tensor_set

    def matrix2tensor(self, inp_matrix, cutoff=True):
        """
        Utilize the matrix to tensor format with or without cutoff
        :param inp_matrix: the input matrix, type=list
        :param cutoff: weather cut of not, type = bool
        :return: the truncated of not mps format of input matrix
        """
        tensor_set = self.get_tensor_set(inp_matrix)
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p, q, lamda_set, lamda_set_value = self.gauge_aux_p_q(left_canonical_tensor, right_canonical_tensor)
        tensor_set = self.tensor_canonical(tensor_set, p, q)
        if cutoff != False:
            tensor_set = self.truncated_tensor(tensor_set)
        return tensor_set, lamda_set, lamda_set_value

    def bi_canonical(self, tensor_set):
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p, q, _, _ = self.gauge_aux_p_q(left_canonical_tensor, right_canonical_tensor)
        tensor_set = self.tensor_canonical(tensor_set, p, q)

        return tensor_set

    def tensor2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an tensor format to matrix format
        :param tensor_set: the input tensor format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape) - 1], [0]))
        # Squeeze the first and the last 1 dimension
        t = t.squeeze(0)
        t = t.squeeze(-1)
        # Caculate the new index for tensor
        tmp1 = torch.tensor(range(len(self.tensor_output_shape))) * 2
        tmp2 = tmp1 + 1
        new_index = torch.cat((tmp1, tmp2), 0)
        # Transpose and reshape to output
        t = t.permute(tuple(new_index))
        t = t.reshape(torch.prod(torch.tensor(self.tensor_input_shape)), torch.prod(torch.tensor(self.tensor_output_shape)))
        return t

    def calculate_total_tensor_param(self, cutoff=True):
        # print("use cutoff: ", cutoff)
        total_size = 0
        if cutoff:
            rank = self.tensor_truncate_ranks
        else:
            rank = self.tensor_ranks
        for i in range(len(self.tensor_input_shape)):
            total_size += rank[i] * self.tensor_input_shape[i] * self.tensor_output_shape[i] * rank[i + 1]

        return total_size

    def new_tensor2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an tensor format to matrix format
        :param tensor_set: the input tensor format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape) - 1], [0]))
        t = t.reshape(torch.prod(torch.tensor(self.tensor_input_shape)), torch.prod(torch.tensor(self.tensor_output_shape)))
        return t

    @staticmethod
    def test_difference(matrix1, matrix2):
        """
        we input an matrix , return the difference between those two matrix
        :param matrix:
        :return:
        """
        v = matrix1 - matrix2
        error = np.linalg.norm(v)
        return error


class LinearDecomTensor(nn.Module):
    '''
    compress using Tensor method
    ref: Compressing deep neural networks by matrix product operators
    '''
    def __init__(self, tensor_input_shape, tensor_output_shape, trunc_num,
        tensor_learn=False,
        CT_learn=False,
        use_bias = True,
        activation = None,
        bias_initializer = 'zeros',
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        debug = False,
        init_seed = 11111986,
        use_dropout=False,
        use_layernorm=False,
        *args,
        **kwargs
    ):
        super(LinearDecomTensor, self).__init__()
        self.trunc_num = trunc_num
        self.tensor_learn = tensor_learn
        self.CT_learn = CT_learn
        tensor_input_shape = np.array(tensor_input_shape)
        tensor_output_shape = np.array(tensor_output_shape)
        #tensor_ranks = np.array(tensor_ranks)
        self.tensor_input_shape = tensor_input_shape
        self.tensor_output_shape = tensor_output_shape
        ##self.tensor_ranks = tensor_ranks
        self.num_dim = tensor_input_shape.shape[0]  # length of the train
        self.use_bias = use_bias
        self.activation = activation
        self.kernel = None
        self.use_dropout = use_dropout
        self.use_layernorm = use_layernorm
        self.tensor_set = None

        self.debug = debug
        self.init_seed = init_seed
        self.lora_linear = False

    def _compute_adapted_weight(self, scaling=None):
        tensor = Tensor(self.tensor_input_shape, self.tensor_output_shape, self.trunc_num)

        weight = tensor.tensor2matrix(self.tensor_set)
        # Merge the weights and mark it

        if self.lora.cotensorsition_mode == "scale":
            delta_w = self.lora.lora_B
        else:
            delta_w = self.lora.lora_B @ self.lora.lora_A
        weight = self.lora.com(weight, delta_w, scaling=scaling)

        return weight

    def build_model(self,input_shape, use_kernel=None, bias=None):
        num_inputs = int(np.prod(input_shape[1::]))


        total_length = torch.from_numpy(np.array(np.sum(self.tensor_input_shape * self.tensor_output_shape *
                              self.tensor_ranks[1:] * self.tensor_ranks[:-1])))
        if isinstance(use_kernel,torch.Tensor):
            self.kernel = torch.empty(size=(total_length,),requires_grad=True,device=torch.device('cuda'))
            self.kernel.data.copy_(use_kernel)
        else:
            self.kernel = torch.empty(size=(total_length,), requires_grad=True, device=torch.device('cuda'))
        self.kernel.contiguous()

        if self.use_bias:
            self.bias = torch.empty(torch.from_numpy(np.array(np.prod(self.tensor_output_shape))), requires_grad=True,
                                    device=torch.device('cuda'))
            if isinstance(bias, torch.Tensor):
                self.bias.data.copy_(bias)
            else:
                nn.init.constant_(self.bias, 0.)

        # Pre-calculate the indices, shapes and cores
        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None] * self.num_dim
        for k in tqdm(range(self.num_dim - 1, -1, -1)):
            # This is the shape of (m_k * r_{k+1}) * (r_k * n_k)
            self.shapes[k] = (self.tensor_input_shape[k] * self.tensor_ranks[k + 1],
                              self.tensor_ranks[k] * self.tensor_output_shape[k])
            # Note that self.cores store only the pointers to the parameter vector
            self.cores[k] = nn.Parameter(data=self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])])
            if 0 < k:  # < self.num_dim-1:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])
        if self.debug:
            print('self.shapes = ' + str(self.shapes))

        # Calculate and print the compression factor
        self.Tensor_size = total_length
        self.full_size = (np.prod(self.tensor_input_shape) * np.prod(self.tensor_output_shape))
        self.compress_factor = 1. * self.Tensor_size / self.full_size
        print('Compression factor = ' + str(self.Tensor_size) + ' / ' \
              + str(self.full_size) + ' = ' + str(self.compress_factor))
    def get_weight(self):
        tensor = Tensor(self.tensor_input_shape, self.tensor_output_shape, self.trunc_num)

        return tensor.tensor2matrix(self.tensor_set)
    def forward(self, x):
        ##################### use rebulid
        tensor = Tensor(self.tensor_input_shape, self.tensor_output_shape, self.trunc_num)
        res = x.reshape(-1, x.shape[-1])
        if self.lora_linear:
            res = F.linear(res, self._compute_adapted_weight(),self.bias)
        else:
            res = F.linear(res, tensor.tensor2matrix(self.tensor_set),self.bias)
        ##################### use rebuild
        ori_shape=x.shape

        return res.view((tuple(ori_shape[:-1])+(-1,)))
        # return res
    def from_pretrained(self, input_shape,kernel_pretrain,tensor_set,bias=None,use_bias=True, device=None):
        if device:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i).to(device)) for i in tensor_set])
        else:
            self.tensor_set = torch.nn.ParameterList([nn.Parameter(torch.from_numpy(i)) for i in tensor_set])
        CT_index = int((len(self.tensor_set)-1)/2)
        if self.tensor_learn:
            # self.tensor_set[3].requires_grad = False
            self.tensor_set[CT_index].requires_grad = False
            # self.tensor_set[5].requires_grad = False
        elif self.CT_learn:
            for i in range(len(self.tensor_set)):
                self.tensor_set[i].requires_grad = False    
            self.tensor_set[CT_index].requires_grad = True    

        if use_bias:
            self.bias = bias
        else:
            logger.info("Check no bias")
            self.bias = None


    def step_trunc(self, tensor_set):
        self.tensor_set = tensor_set

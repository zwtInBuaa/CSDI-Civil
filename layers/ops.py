import torch
import torch.nn.functional as F
from einops import reduce
from torch.autograd import Variable


def reverse_tensor(tensor=None, axis=-1):
    if tensor is None:
        return None
    if tensor.dim() <= 1:
        return tensor
    indices = range(tensor.size()[axis])[::-1]
    indices = Variable(torch.LongTensor(indices), requires_grad=False).to(tensor.device)
    return tensor.index_select(axis, indices)
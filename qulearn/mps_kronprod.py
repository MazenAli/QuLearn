import tntorch
import torch

from .types import MPS


def kron(tleft: MPS, tright: MPS) -> MPS:
    """
    Performs the Kronecker product of two MPS tensors.

    :param tleft: The first MPS tensor.
    :type tleft: MPS
    :param tright: The second MPS tensor.
    :type tright: MPS
    :return: The MPS tensor resulting from the Kronecker product of `tleft` and `tright`.
    :rtype: MPS
    """
    c1 = tleft.cores
    c2 = tright.cores
    c3 = c1 + c2
    t3 = tntorch.Tensor(c3)

    return t3


def zkron(tleft: MPS, tright: MPS) -> MPS:
    """
    Performs the z-ordered Kronecker product of two MPS tensors.
    See https://arxiv.org/abs/1802.02839.

    :param tleft: The first MPS tensor.
    :type tleft: MPS
    :param tright: The second MPS tensor.
    :type tright: MPS
    :return: The MPS tensor resulting from the Kronecker product of `tleft` and `tright`.
    :rtype: MPS
    """
    _core_length_check(tleft, tright)

    coresleft = tleft.cores
    coresright = tright.cores

    if len(coresleft) != len(coresright):
        raise ValueError("The number of cores in the left and right MPS must be the same.")

    coresout = []

    for i in range(len(coresleft)):
        coreleft = coresleft[i]
        coreright = coresright[i]
        rankleft1 = coreleft.shape[0]
        rankleft2 = coreleft.shape[-1]
        rankright1 = coreright.shape[0]
        rankright2 = coreright.shape[-1]

        site_dim = coreleft.shape[1]
        core = torch.empty((rankleft1 * rankright1, site_dim, rankleft2 * rankright1))
        for k in range(site_dim):
            core[:, k, :] = torch.kron(coreleft[:, k, :], torch.eye(rankright1))
        coresout.append(core)

        site_dim = coreright.shape[1]
        core = torch.empty((rankleft2 * rankright1, site_dim, rankleft2 * rankright2))
        for k in range(site_dim):
            core[:, k, :] = torch.kron(torch.eye(rankleft2), coreright[:, k, :])
        coresout.append(core)

    tout = tntorch.Tensor(coresout)
    return tout


def zkron_joined(tleft, tright):
    """
    Performs the z-ordered Kronecker product of two MPS tensors,
    and joins the physical indices of tleft and tright into one.
    See https://arxiv.org/abs/1802.02839.

    :param tleft: The first MPS tensor.
    :type tleft: MPS
    :param tright: The second MPS tensor.
    :type tright: MPS
    :return: The MPS tensor resulting from the Kronecker product of `tleft` and `tright`.
    :rtype: MPS
    """
    _core_length_check(tleft, tright)

    c1 = tleft.cores
    c2 = tright.cores
    c3 = [torch.kron(a, b) for a, b in zip(c1, c2)]

    t3 = tntorch.Tensor(c3)
    return t3


def _core_length_check(tleft, tright):
    coresleft = tleft.cores
    coresright = tright.cores
    if len(coresleft) != len(coresright):
        raise ValueError("The number of cores in the left and right MPS must be the same.")

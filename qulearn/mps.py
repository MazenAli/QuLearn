# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from typing import List

import math
import torch
import tntorch
from qulearn.hat_basis import HatBasis

MPS: TypeAlias = tntorch.tensor.Tensor
Tensor: TypeAlias = torch.Tensor


class MPSQGates:
    """
    Converts Matrix Product States (MPS) to quantum gates.

    :param mps: The MPS from which quantum gates will be extracted.
    :type mps: MPS
    """

    def __init__(self, mps: MPS) -> None:
        self.mps = mps
        self.max_rank_power = compute_max_rank_power(mps)

    def qgates(self) -> List[Tensor]:
        """
        Extracts quantum gates from the MPS.

        :returns: A list of Tensors, each representing a unitary matrix for a quantum gate.
        :rtype: List[Tensor]
        """

        N = self.mps.dim()
        mps = self.pad_cores()
        mps.orthogonalize(N - 1)

        Us = []
        core = self.contract(mps=mps, L=self.max_rank_power + 1)
        Q0 = self.left_core_reshape(core)
        Us.append(embed2unitary(Q0))

        cores = mps.cores.copy()
        for j in range(self.max_rank_power + 1, N):
            core = cores[j]
            Q = self.reg_core_reshape(core)
            Us.append(embed2unitary(Q))

        return Us

    def pad_cores(self) -> MPS:
        """
        Pads the cores of the MPS to match the maximum rank, ensuring uniform core dimensions.

        :returns: A new MPS with padded cores.
        :rtype: MPS
        """

        chi = 2**self.max_rank_power
        cores = self.mps.cores.copy()

        core = cores[0]
        rankR = chi - core.shape[-1]
        pad_first = (0, rankR)
        cores[0] = torch.nn.functional.pad(core, pad_first)

        core = cores[-1]
        rankL = chi - core.shape[0]
        pad_last = (0, 0, 0, 0, 0, rankL)
        cores[-1] = torch.nn.functional.pad(core, pad_last)

        for j in range(1, len(cores) - 1):
            core = cores[j]
            rankL = chi - core.shape[0]
            rankR = chi - core.shape[-1]
            pad = (0, rankR, 0, 0, 0, rankL)
            cores[j] = torch.nn.functional.pad(core, pad)

        mps = tntorch.Tensor(cores)

        return mps

    def contract(self, mps, L: int) -> Tensor:
        """
        Contracts the first L cores from the left of the MPS to form a single tensor.

        :param mps: The MPS from which quantum gates will be extracted.
        :type mps: MPS
        :param L: The number of cores to contract.
        :type L: int
        :returns: A Tensor resulting from the contraction of the first L cores.
        :rtype: Tensor
        """

        cores = mps.cores
        result = cores[0].clone().detach()
        for i in range(1, L):
            result = torch.tensordot(result, cores[i], dims=([result.dim() - 1], [0]))

        return result

    def left_core_reshape(self, core: Tensor) -> Tensor:
        """
        Reshapes a core tensor for the left-most core, preparing it for SVD and embedding into a unitary matrix.

        :param core: The core tensor to reshape.
        :type core: Tensor
        :returns: The reshaped core tensor.
        :rtype: Tensor
        """

        rows = list(range(0, self.max_rank_power + 2))
        row_size = int(torch.prod(torch.tensor(core.shape)[rows]).item())
        Q = core.reshape(row_size, -1)
        return Q

    def reg_core_reshape(self, core: Tensor) -> Tensor:
        """
        Reshapes a regular core tensor, preparing it for SVD and embedding into a unitary matrix.

        :param core: The core tensor to reshape.
        :type core: Tensor
        :returns: The reshaped core tensor.
        :rtype: Tensor
        """

        rows = [0, 1]
        row_size = int(torch.prod(torch.tensor(core.shape)[rows]).item())
        Q = core.reshape(row_size, -1)
        return Q


class HatBasisMPS:
    """
    Generates Matrix Product States (MPS) corresponding to evaluations of linear hat basis functions.

    :param basis: The hat basis to use for generating the MPS.
    :type basis: HatBasis

    .. note::
       The number of nodes in the hat basis must be a power of 2, corresponding to the number of qubits used.
       Currently works only for scalar inputs x.
    """

    def __init__(self, basis: HatBasis) -> None:
        self.basis = basis

        num_qubits = math.log2(basis.num_nodes)
        if not num_qubits.is_integer():
            raise ValueError(
                f"Number of nodes ({basis.num_nodes}) " "must be a power of 2."
            )

        self.num_sites = int(num_qubits)

    def __call__(self, x: Tensor) -> MPS:
        """
        Constructs the MPS of the hat basis evaluated at a given point x.

        :param x: The input at which to evaluate the hat basis.
        :type x: Tensor
        :returns: The MPS at point x.
        :rtype: MPS
        """

        return self.eval(x)

    def eval(self, x: Tensor) -> MPS:
        """
        Constructs the MPS of the hat basis evaluated at a given point x.

        :param x: The input at which to evaluate the hat basis.
        :type x: Tensor
        :returns: The MPS at point x.
        :rtype: MPS
        """

        position = self.basis.position(x)
        first, second = self.basis.nonz_vals(x)

        # works only for scalars
        idx = int(position.item())
        a = first.item()
        b = second.item()

        mps = self.mps_hatbasis(a, b, idx)

        return mps

    def mps_hatbasis(self, first: float, second: float, idx: int) -> MPS:
        """
        Generates an MPS for the hat basis vector.

        :param first: The first non-zero value in the hat basis function.
        :type first: float
        :param second: The second non-zero value in the hat basis function.
        :type second: float
        :param idx: The index of the hat basis function.
        :type idx: int
        :returns: An MPS representing the hat basis.
        :rtype: MPS
        """

        even = idx % 2 == 0
        if even:
            mps = self.mps_hatbasis_evenidx(first, second, idx)
        else:
            mps = self.mps_hatbasis_oddidx(first, second, idx)

        return mps

    def mps_hatbasis_evenidx(self, first: float, second: float, idx: int) -> MPS:
        """
        Generates an MPS for an even index in the hat basis.

        :param first: The first non-zero value in the hat basis function.
        :type first: float
        :param second: The second non-zero value in the hat basis function.
        :type second: float
        :param idx: The index of the hat basis function.
        :type idx: int
        :returns: An MPS representing the hat basis.
        :rtype: MPS
        """
        binidx = format(idx, "0{}b".format(self.num_sites))

        cores = []
        for j in range(self.num_sites - 1):
            k = int(binidx[j])
            core = torch.zeros((1, 2, 1))
            core[0, k, 0] = 1.0
            cores.append(core)

        core = torch.zeros(1, 2, 1)
        core[0, 0, 0] = first
        core[0, 1, 0] = second
        cores.append(core)
        mps = tntorch.Tensor(cores)

        return mps

    def mps_hatbasis_oddidx(self, first: float, second: float, idx: int) -> MPS:
        """
        Generates an MPS for an odd index in the hat basis.

        :param first: The first non-zero value in the hat basis function.
        :type first: float
        :param second: The second non-zero value in the hat basis function.
        :type second: float
        :param idx: The index of the hat basis function.
        :type idx: int
        :returns: An MPS representing the hat basis.
        :rtype: MPS
        """

        binidx = format(idx, "0{}b".format(self.num_sites))
        zerobit = zerobit_position_odd(idx, self.num_sites)

        cores = []
        for j in range(zerobit):
            k = int(binidx[j])
            core = torch.zeros((1, 2, 1))
            core[0, k, 0] = 1.0
            cores.append(core)

        if zerobit != self.num_sites - 1:
            core = torch.zeros(1, 2, 2)
            core[0, 0, 1] = 1.0
            core[0, 1, 0] = 1.0
            cores.append(core)

        for j in range(zerobit + 1, self.num_sites - 1):
            core = torch.zeros((2, 2, 2))
            core[0, 0, 0] = 1.0
            core[1, 1, 1] = 1.0
            cores.append(core)

        if zerobit != self.num_sites - 1:
            core = torch.zeros(2, 2, 1)
            core[0, 0, 0] = second
            core[1, 1, 0] = first
            cores.append(core)
        else:
            core = torch.zeros(1, 2, 1)
            core[0, 0, 0] = second
            core[0, 1, 0] = first
            cores.append(core)

        mps = tntorch.Tensor(cores)

        return mps


def compute_max_rank_power(mps: MPS) -> int:
    """
    Computes the power in base 2 of the maximum rank of an MPS tensor.

    :param mps: An MPS tensor whose maximum rank is to be computed.
    :type mps: MPS
    :return: The maximum rank power as an integer.
    :rtype: int
    """

    s = int(torch.ceil(torch.log2(max(mps.ranks_tt))))
    return s


def embed2unitary(Q: Tensor) -> Tensor:
    """
    Embeds a matrix Q into a unitary matrix by appending orthonormal columns.

    :param Q: The input matrix to be embedded into a unitary matrix.
              Its shape is assumed to be (m, k), where m >= k.
    :type Q: Tensor
    :return: The unitary matrix formed by appending orthonormal columns to Q.
    :rtype: Tensor
    """

    k = Q.shape[-1]
    U, _, _ = torch.linalg.svd(Q, full_matrices=True)
    Q_ = torch.cat((Q, U[:, k:]), dim=1)
    return Q_


def zerobit_position_odd(k, n):
    """
    Finds the position of the most least zero bit in the binary representation of an odd integer.

    :param k: An odd integer.
    :type k: int
    :param n: The total length for the binary representation.
    :type n: int
    :return: The position of the least significant zero bit.
    :rtype: int
    """

    if k % 2 == 0:
        m = 0
    else:
        m = int(math.log2((-k) & (k + 1)))
    pos = n - m - 1
    return int(pos)

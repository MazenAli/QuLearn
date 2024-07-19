from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from abc import ABC, abstractmethod
from itertools import product

import numpy as np
import torch
from scipy.stats import qmc
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset

Tensor: TypeAlias = torch.Tensor
Array: TypeAlias = np.ndarray
Device: TypeAlias = torch.device
DataOut: TypeAlias = Dict[str, Tensor]
Loader: TypeAlias = DataLoader
Model: TypeAlias = Module
ParameterList: TypeAlias = List[List[Tensor]]
CDevice: TypeAlias = torch.device
DType: TypeAlias = torch.dtype
D = TypeVar("D")
L = TypeVar("L")


class DataGenTorch(ABC, Generic[D, L]):
    """
    Abstract base class for generating data in PyTorch.

    :param batch_size: Size of batch for data loader. Defaults to None (batch mode).
    :type batch_size: int
    :param shuffle: Shuffle data in loader. Defaults to False.
    :type shuffle: bool
    :param seed: The seed used to initialize the random number generator.
        Defaults to None.
    :type seed: int
    :param device: The device where to run the computations. Defaults to
        torch.device("cpu").
    :type device: Device
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        device: Device = torch.device("cpu"),
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device

    @abstractmethod
    def gen_data(self, *args, **kwargs) -> D:
        """
        Generate the data.

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        :returns: The generated data.
        :rtype: D
        """
        pass

    @abstractmethod
    def data_to_loader(self, *args, **kwargs) -> L:
        """
        Convert data from gen_data to pytorch data loader.

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        :returns: Data loader.
        :rtype: L
        """
        pass


class PriorTorch(ABC, Generic[D]):
    """
    Abstract base class for generating priors in PyTorch.

    :param sizex: Dimension of feature space.
    :type sizex: int
    :param seed: The seed used to initialize the random number
        generator. Defaults to None.
    :type seed: int, optional
    :param device: The device where to run the computations. Defaults to
        torch.device("cpu").
    :type device: Device
    """

    def __init__(
        self,
        sizex: int,
        seed: Optional[int] = None,
        device: Device = torch.device("cpu"),
    ) -> None:
        self.sizex = sizex
        self.seed = seed
        self.device = device

    @abstractmethod
    def gen_data(self, *args, **kwargs) -> D:
        """
        Generate the data.

        :param args: Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        :returns: The generated data.
        :rtype: D
        """
        pass


class DataGenCapacity(DataGenTorch[DataOut, Loader]):
    """
    Generates data for memory capacity estimation.

    :param sizex: The size of the input data.
    :type sizex: int
    :param num_samples: The number of output samples to generate.
        Defaults to 10.
    :type num_samples: int, optional
    :param scale: The re-scaling factor for uniform random numbers in [0, 1].
        Defaults to 2.0.
    :type scale: float, optional
    :param shift: The shift value for uniform random numbers in [0, 1].
        Defaults to -1.0.
    :type shift: float, optional
    :param kwargs: Keyword arguments passed to the base class.
    """

    def __init__(
        self,
        sizex: int,
        num_samples: int = 10,
        scale: float = 2.0,
        shift: float = -1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.sizex = sizex
        self.num_samples = num_samples
        self.scale = scale
        self.shift = shift

    def gen_data(self, N: int) -> DataOut:
        """
        Generate the data.

        :param N: The number of inputs for the model.
        :type N: int
        :return: A dictionary containing the generated data.
        :rtype: DataOut
        """

        X, Y = gen_dataset_capacity(
            N=N,
            sizex=self.sizex,
            num_samples=self.num_samples,
            seed=self.seed,
            scale=self.scale,
            shift=self.shift,
            device=self.device,
        )

        data = {"X": X, "Y": Y}

        return data

    def data_to_loader(self, data: DataOut, s: int) -> Loader:
        """
        Convert data to pytorch loader.

        :param data: Output of gen_data.
        :type data: DataOut
        :param s: Current label sample.
        :type s: int
        :rtype: Loader
        :raises ValueError: For invalid data or index s.
        """
        self._check_data(data)

        Ny = data["Y"].size(0)
        if s < 0 or s >= Ny:
            raise ValueError(f"Index i ({s}) must be between 0 and Ny ({Ny}).")

        X = data["X"]
        Y = data["Y"][s]
        Nx = X.size(0)
        dataset = TensorDataset(X, Y)
        batch_size = Nx
        if self.batch_size is not None:
            batch_size = self.batch_size

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)

        return loader

    def _check_data(self, data: DataOut):
        if "X" not in data:
            raise ValueError("Data does not contain X key")

        if "Y" not in data:
            raise ValueError("Data does not contain Y key")

        X = data["X"]
        Y = data["Y"]

        check = len(X.shape)
        if check != 2:
            raise ValueError(f"X must be 2-dim (not {check})")

        check = len(Y.shape)
        if len(Y.shape) != 3:
            raise ValueError(f"Y must be 3-dim (not {check})")


class DataGenFat(DataGenTorch[DataOut, Loader]):
    """
    Generates data for estimating fat shattering dimension.

    :param prior: Data generator for prior X.
    :type prior: DataGenTorch
    :param Sb: The number of binary samples to check shattering.
        Defaults to 10.
    :type Sb: int, optional
    :param Sr: The number of level offset samples to check shattering.
        Defaults to 10.
    :type Sr: int, optional
    :param gamma: The fat shattering parameter gamma.
        Defaults to 0.0 (pseudo-dimension).
    :type gamma: float, optional
    :param kwargs: Keyword arguments passed to the base class.
    """

    def __init__(
        self,
        prior: PriorTorch,
        Sb: int = 10,
        Sr: int = 10,
        gamma: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.Sb = Sb
        self.Sr = Sr
        self.gamma = gamma
        self.prior = prior

    def gen_data(self, d: int) -> DataOut:
        """
        Generate the data.

        :param d: The number of inputs to shatter.
        :type d: int
        :return: A dictionary containing the generated data.
        :rtype: DataOut
        """

        X = self.prior.gen_data(d)
        b = generate_samples_b_fat(d=d, S=self.Sb, seed=self.seed)
        r = generate_samples_r_fat(d=d, S=self.Sr, seed=self.seed)
        Y = gen_synthetic_labels_fat(b, r, self.gamma, self.device)

        data = {"X": X, "Y": Y, "b": b, "r": r}

        return data

    def data_to_loader(self, data: DataOut, sr: int, sb: int) -> Loader:
        """
        Convert data to pytorch loader.

        :param data: Output of gen_data.
        :type data: DataOut
        :param sr: Current r sample.
        :type sr: int
        :param sb: Current b sample.
        :type sb: int
        :returns: Pytorch data loader.
        :rtype: Loader
        :raises ValueError: For invalid data or indeces sr or sb.
        """
        self._check_data(data)

        Nr = data["Y"].size(0)
        Nb = data["Y"].size(1)
        if sr < 0 or sr >= Nr:
            raise ValueError(f"Index sr ({sr}) must be between 0 and Ny ({Nr}).")
        if sb < 0 or sb >= Nb:
            raise ValueError(f"Index sb ({sb}) must be between 0 and Ny ({Nb}).")

        X = data["X"]
        Nx = X.size(0)
        Y = data["Y"][sr, sb]
        dataset = TensorDataset(X, Y)
        batch_size = Nx
        if self.batch_size is not None:
            batch_size = self.batch_size

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)

        return loader

    def _check_data(self, data: DataOut):
        if "X" not in data:
            raise ValueError("Data does not contain X key")

        if "Y" not in data:
            raise ValueError("Data does not contain Y key")

        X = data["X"]
        Y = data["Y"]

        check = len(X.shape)
        if check != 2:
            raise ValueError(f"X must be 2-dim (not {check})")

        check = len(Y.shape)
        if check != 4:
            raise ValueError(f"Y must be 4-dim (not {check})")


class DataGenRademacher(DataGenTorch[DataOut, Loader]):
    """
    Generates uniform data for estimating the empirical Rademacher complexity.

    :param prior: Prior for generating X samples.
    :type prior: PriorTorch
    :param num_sigma_samples: Number of samples for sigma.
            Defaults to 10.
    :type num_sigma_samples: int, optional
    :param num_data_samples: Number of samples for data sets.
            Defaults to 10.
    :type num_data_samples: int, optional
    :param gamma: The fat shattering parameter gamma
        Defaults to 0.0 (pseudo-dimension).
    :type gamma: float, optional
    :param scale: The re-scaling factor for uniform random numbers in [0, 1].
        Defaults to 2.0.
    :type scale: float, optional
    :param shift: The shift value for uniform random numbers in [0, 1].
        Defaults to -1.0.
    :type shift: float, optional
    :param kwargs: Keyword arguments passed to the base class.
    """

    def __init__(
        self,
        prior: PriorTorch,
        num_sigma_samples: int = 10,
        num_data_samples: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.prior = prior
        self.num_sigma_samples = num_sigma_samples
        self.num_data_samples = num_data_samples

    def gen_data(self, m: int) -> DataOut:
        """Generate the data.

        :param m: Size of data set.
        :type m: int
        :param m: int:
        :return: A dictionary containing the generated data.
        :rtype: DataOut
        """

        X = self.prior.gen_data(m * self.num_data_samples)
        X = torch.reshape(X, (self.num_data_samples, m, self.prior.sizex))
        sigmas = gen_sigmas(m=m * self.num_sigma_samples, seed=self.seed, device=self.device)
        sigmas = torch.reshape(sigmas, (self.num_sigma_samples, m))

        data = {"X": X, "sigmas": sigmas}

        return data

    def data_to_loader(self, data: DataOut, s: int) -> Loader:
        """
        Convert data to pytorch loader.

        :param data: Output of gen_data.
        :type data: DataOut
        :param s: Current sample.
        :type s: int
        :return: Pytorch data loader.
        :rtype: Loader
        :raises ValueError: For invalid data or index s.
        """
        self._check_data(data)

        Ns = data["X"].size(0)
        if s < 0 or s >= Ns:
            raise ValueError(f"Index s ({s}) must be between 0 and Ns ({Ns}).")

        X = data["X"][s]
        Nx = X.size(0)
        Y = torch.zeros(Nx, device=X.device)
        dataset = TensorDataset(X, Y)
        batch_size = Nx
        if self.batch_size is not None:
            batch_size = self.batch_size

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=self.shuffle)

        return loader

    def _check_data(self, data: DataOut):
        if "X" not in data:
            raise ValueError("Data does not contain X key")

        X = data["X"]
        check = len(X.shape)

        if check != 3:
            raise ValueError(f"X must be 3-dim (not {check})")


class UniformPrior(PriorTorch[Tensor]):
    """
    Generates uniform prior X.

    :param sizex: The size of the input data (dim of feature space).
    :type sizex: int
    :param scale: The re-scaling factor for uniform random numbers in [0, 1].
        Defaults to 2.0.
    :type scale: float, optional
    :param shift: The shift value for uniform random numbers in [0, 1].
        Defaults to -1.0.
    :type shift: float, optional
    :param kwargs: Keyword arguments passed to the base class.
    """

    def __init__(self, sizex: int, scale: float = 2.0, shift: float = -1.0, **kwargs) -> None:
        super().__init__(sizex, **kwargs)

        self.scale = scale
        self.shift = shift

    def gen_data(self, m: int) -> Tensor:
        """
        Generate the data.

        :param m: Size of data set.
        :type m: int
        :return: Prior X.
        :rtype: Tensor
        """

        X = gen_synthetic_features(
            d=m,
            sizex=self.sizex,
            seed=self.seed,
            scale=self.scale,
            shift=self.shift,
            device=self.device,
        )

        return X


class NormalPrior(PriorTorch[Tensor]):
    """
    Generates normal prior for X.

    :param sizex: The size of the input data (dim of feature space).
    :type sizex: int
    :param scale: The re-scaling factor for standard normal.
        Defaults to 1.0.
    :type scale: float, optional
    :param shift: The shift value for standard normal.
            Defaults to 0.0.
    :type shift: float, optional
    :param kwargs: Keyword arguments passed to the base class.
    """

    def __init__(self, sizex: int, scale: float = 1.0, shift: float = 0.0, **kwargs) -> None:
        super().__init__(sizex, **kwargs)

        self.scale = scale
        self.shift = shift

    def gen_data(self, m: int) -> Tensor:
        """
        Generate the data.

        :param m: Size of data set.
        :type m: int
        :return: Prior X.
        :rtype: Tensor
        """

        X = gen_synthetic_features_normal(
            d=m,
            sizex=self.sizex,
            seed=self.seed,
            scale=self.scale,
            shift=self.shift,
            device=self.device,
        )

        return X


def gen_dataset_capacity(
    N: int,
    sizex: int,
    num_samples: int = 10,
    seed: Optional[int] = None,
    scale: float = 2.0,
    shift: float = -1.0,
    device: Device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor]:
    """
    Generates a dataset of input features x and output labels y.

    :param N: The number of inputs.
    :type N: int
    :param sizex: The dimension of each input.
    :type sizex: int
    :param num_samples: The number of output samples to generate. Defaults to 10.
    :type num_samples: int, optional
    :param seed: The random seed to use for generating the dataset.
        Defaults to None.
    :type seed: int, optional
    :param scale: The re-scaling factor for uniform random numbers in
        [0,1]. Defaults to 2.0.
    :type scale: float, optional
    :param shift: The shift value for uniform random numbers [0,1].
        Defaults to -1.0.
    :type shift: float, optional
    :param device: Torch device to run on. Defaults to CPU.
    :type device: Device
    :return: A tuple containing the input tensor x of shape (N, sizex)
        and the output (num_samples, N).
    :rtype: Tuple[Tensor, Tensor]
    """

    seed_ = seed
    if seed_ is None:
        seed_ = torch.seed()

    generator = torch.manual_seed(seed_)

    x = (
        scale
        * torch.rand(
            N,
            sizex,
            dtype=torch.float64,
            device=device,
            requires_grad=False,
            generator=generator,
        )
        + shift
    )
    y = (
        scale
        * torch.rand(
            num_samples,
            N,
            1,
            dtype=torch.float64,
            device=device,
            requires_grad=False,
            generator=generator,
        )
        + shift
    )

    return x, y


def generate_samples_b_fat(
    d: int,
    S: int,
    seed: Optional[int] = None,
) -> Array:
    """
    Generate S unique samples of b from {0, 1}^d.

    :param d: Number of input data samples for shattering.
    :type d: int
    :param S: Number of binary samples to check shattering (for
        scalability if d is large).
    :type S: int
    :param seed: The random seed. Defaults to None.
    :type seed: int, optional
    :return: An array of shape (S, d) containing unique samples of b.
    :rtype: Array
    """

    rng = np.random.default_rng(seed)
    max_values = 2**d
    if S >= max_values:
        possible_values = list(product([0, 1], repeat=d))
        return np.array(possible_values)
    else:
        samples: Set[Tuple[int, ...]] = set()
        while len(samples) < S:
            b_sample = tuple(rng.integers(0, 2, size=d))
            samples.add(b_sample)
        return np.array(list(samples))


def generate_samples_r_fat(
    d: int,
    S: int,
    seed: Optional[int] = None,
) -> Array:
    """
    Generate S samples of r from [0, 1]^d using Latin Hypercube Sampling.

    :param d: Dimension of the feature space.
    :type d: int
    :param S: Number of samples to generate.
    :type S: int
    :param seed: The random seed. Defaults to None.
    :type seed: int, optional
    :return: An array of shape (S, d) containing samples of r.
    :rtype: Array
    """

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    r_samples = sampler.random(n=S)
    return r_samples


def gen_synthetic_features(
    d: int,
    sizex: int,
    seed: Optional[int] = None,
    scale: float = 2.0,
    shift: float = -1.0,
    device: Device = torch.device("cpu"),
) -> Tensor:
    """
    Generates d inputs x of dimension sizex sampled uniformly from scale*[0,1]+shift.

    :param d: The number of inputs x to generate.
    :type d: int
    :param sizex: The size of each input.
    :type sizex: int
    :param seed: The random seed to use for generating the features.
        Defaults to None.
    :type seed: int, optional
    :param scale: The re-scaling factor for uniform random numbers in
        [0,1]. Defaults to 2.0.
    :type scale: float, optional
    :param shift: The shift value for uniform random numbers [0,1].
        Defaults to -1.0.
    :type shift: float, optional
    :param device: Torch device to run on. Defaults to CPU.
    :type device: Device
    :return: Tensor X of shape (d, sizex).
    :rtype: Tensor
    """

    seed_ = seed
    if seed_ is None:
        seed_ = torch.seed()

    generator = torch.manual_seed(seed_)

    x = (
        scale
        * torch.rand(
            d,
            sizex,
            dtype=torch.float64,
            device=device,
            requires_grad=False,
            generator=generator,
        )
        + shift
    )

    return x


def gen_synthetic_features_normal(
    d: int,
    sizex: int,
    seed: Optional[int] = None,
    scale: float = 1.0,
    shift: float = 0.0,
    device: Device = torch.device("cpu"),
) -> Tensor:
    """
    Generates d inputs x of dimension sizex sampled from N(shift, scale^2).

    :param d: The number of inputs x to generate.
    :type d: int
    :param sizex: The size of each input.
    :type sizex: int
    :param seed: The random seed to use for generating the features.
        Defaults to None.
    :type seed: int, optional
    :param scale: The re-scaling factor for uniform random numbers in
        [0,1]. Defaults to 1.0.
    :type scale: float, optional
    :param shift: The shift value for uniform random numbers [0,1].
        Defaults to 0.0.
    :type shift: float, optional
    :param device: Torch device to run on. Defaults to CPU.
    :type device: Device, optional
    :return: Tensor X of shape (d, sizex).
    :rtype: Tensor
    """

    seed_ = seed
    if seed_ is None:
        seed_ = torch.seed()

    generator = torch.manual_seed(seed_)

    x = (
        scale
        * torch.randn(
            d,
            sizex,
            dtype=torch.float64,
            device=device,
            requires_grad=False,
            generator=generator,
        )
        + shift
    )

    return x


def gen_synthetic_labels_fat(
    b: Array,
    r: Array,
    gamma: float = 0.0,
    device: Device = torch.device("cpu"),
) -> Tensor:
    """
    Generate constant label values equal to r_i + gamma
    when b_i = 1 and r_i - gamma when b_i = 0.

    :param b: An array of shape (Sb,d)
            containing Sb samples of d-dim binary values.
    :type b: Array
    :param r: An array of shape (Sr,d)
            containing Sr samples of d-dim real values in [0, 1].
    :type r: Array
    :param gamma: The fat-shattering margin value.
            Defaults to 0.0.
    :type gamma: float, optional
    :param device: Torch device to run on. Defaults to CPU.
    :type device: Device, optional
    :return: Y of shape (Sr, Sb, d, 1).
    :rtype: Tensor
    :raises ValueError: If the length of b[0] is not the same as the length of r[0].
    """

    Sb = len(b)
    Sr = len(r)
    d = d1 = len(b[0])
    d2 = len(r[0])

    if d1 != d2:
        raise ValueError(
            f"The length of b[0] and r[0] are {d1} and {d2}. " f"Should be constant and the same."
        )

    labels = np.zeros((Sr, Sb, d, 1))
    for sr in range(Sr):
        for sb in range(Sb):
            for i in range(d):
                if b[sb][i] == 1:
                    labels[sr, sb, i, 0] = r[sr, i] + gamma
                else:
                    labels[sr, sb, i, 0] = r[sr, i] - gamma

    y = torch.tensor(labels, dtype=torch.float64, device=device, requires_grad=False)

    return y


def gen_sigmas(m: int, seed: Optional[int] = None, device: Device = torch.device("cpu")) -> Tensor:
    """
    Random vector of +-1.

    :param m: Number of sigmas.
    :type m: int
    :param seed: The random seed to use for generating the features.
        Defaults to None.
    :type seed: int, optional
    :param device: Torch device to run on. Defaults to CPU.
    :type device: Device, optional
    :return: Tensor of sigmas.
    :rtype: Tensor
    """

    seed_ = seed
    if seed_ is None:
        seed_ = torch.seed()

    generator = torch.manual_seed(seed_)

    sigmas = torch.randint(2, (m,), device=device, requires_grad=False, generator=generator) * 2 - 1

    return sigmas


def generate_lhs_samples(
    n_samples: int,
    n_dims: int,
    lower_bound: float,
    upper_bound: float,
    seed: Optional[int] = None,
) -> Array:
    """
    Generate Latin Hypercube Samples (LHS) within the specified range.

    :param n_samples: The number of samples to generate.
    :type n_samples: int
    :param n_dims: The number of dimensions for each sample.
    :type n_dims: int
    :param lower_bound: The lower bound for the samples.
    :type lower_bound: float
    :param upper_bound: The upper bound for the samples.
    :type upper_bound: float
    :param seed: The random seed. Defaults to None.
    :type seed: int, optional
    :return: A numpy array containing the generated samples.
    :rtype: Array
    """

    sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
    sample = sampler.random(n=n_samples)
    return lower_bound + sample * (upper_bound - lower_bound)


def generate_model_lhs_samples(
    model: Model,
    n_samples: int,
    lower_bound: float,
    upper_bound: float,
    device: Optional[CDevice] = None,
    dtype: Optional[DType] = None,
    seed: Optional[int] = None,
) -> ParameterList:
    """
    Generate Latin Hypercube Samples (LHS) for each parameter of a given model.

    :param model: The model whose parameters to sample.
    :type model: Model
    :param n_samples: The number of samples to generate.
    :type n_samples: int
    :param lower_bound: The lower bound for the samples.
    :type lower_bound: float
    :param upper_bound: The upper bound for the samples.
    :type upper_bound: float
    :param cdevice: Classical device to store the parameters, defaults to None.
    :type cdevice: CDevice, optional
    :param dtype: Data type of the parameters, defaults to None.
    :type dtype: DType, optional
    :param seed: The random seed. Defaults to None.
    :type seed: int, optional
    :return: A list of lists containing samples for each parameter.
    :rtype: ParameterList
    """

    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    samples = []
    for p in model_parameters:
        n_dims = p.numel()
        sample_parameter = generate_lhs_samples(
            n_samples=n_samples,
            n_dims=n_dims,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=seed,
        )
        samples.append(sample_parameter.reshape((n_samples,) + p.shape))
    parameter_list = [
        [torch.tensor(samples[j][i], device=device, dtype=dtype) for j in range(len(samples))]
        for i in range(n_samples)
    ]

    return parameter_list

from typing import Optional, TypeVar, Generic, Tuple, Dict, Set, Union
from abc import ABC, abstractmethod
import torch
import numpy as np
from itertools import product
from scipy.stats import qmc

Tensor = Union[torch.Tensor, np.ndarray]
Device = torch.device
D = TypeVar("D")


class DataGenTorch(ABC, Generic[D]):
    """
    Abstract base class for generating data in PyTorch.

    Args:
        seed (int, optional): The seed used to initialize the
            random number generator. Defaults to None.
        device (torch.device, optional): The device where to run the computations.
            Defaults to torch.device("cpu").
    """

    def __init__(
        self, seed: Optional[int] = None, device: Device = torch.device("cpu")
    ) -> None:
        self.__seed = seed

        if self.__seed is not None:
            torch.manual_seed(self.__seed)

        self.device = device

    @abstractmethod
    def gen_data(self, *args, **kwargs) -> D:
        """
        Generate the data.

        Returns:
            D: The generated data.
        """
        pass

    @property
    def seed(self) -> Optional[int]:
        """
        Get the seed used to initialize the random number generator.

        Returns:
            int: The seed used to initialize the random number generator.
        """

        return self.__seed

    @seed.setter
    def seed(self, seed_: int) -> None:
        """
        Set the seed used to initialize the random number generator.

        Args:
            seed_ (int): The seed used to initialize the random number generator.
        """

        self.__seed = seed_
        if self.__seed is not None:
            torch.manual_seed(self.__seed)


class DataGenCapacity(DataGenTorch[Dict[str, Tensor]]):
    """
    Generates data for capacity estimation.

    Args:
        sizex (int): The size of the input data.
        scale (float, optional): The re-scaling factor for uniform
            random numbers in [0, 1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform
            random numbers in [0, 1]. Defaults to -1.0.
        args: Variable length argument list passed to the base class.
        kwargs: Arbitrary keyword arguments passed to the base class.
    """

    def __init__(
        self, sizex: int, scale: float = 2.0, shift: float = -1.0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.sizex = sizex
        self.scale = scale
        self.shift = shift

    def gen_data(self, N: int, num_samples: int = 10) -> Dict[str, Tensor]:
        """
        Generate the data.

        Args:
            N (int): The number of inputs for the QNN.
            num_samples (int): The number of output samples to generate. Defaults to 10.

        Returns:
            data (Dict[str, Tensor]): A dictionary containing the generated data.
        """

        X, Y = gen_dataset_capacity(
            N,
            self.sizex,
            num_samples,
            self.seed,
            self.scale,
            self.shift,
            self.device,
        )

        data = {"X": X, "Y": Y}

        return data


class DataGenFat(DataGenTorch[Dict[str, Tensor]]):
    """
    Generates data for estimating fat shattering dimension.

    Args:
        sizex (int): The size of the input data (dim of feature space).
        gamma (float): The fat shattering parameter gamma.
            Defaults to 0.0 (pseudo-dimension).
        scale (float, optional): The re-scaling factor for uniform
            random numbers in [0, 1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform
            random numbers in [0, 1]. Defaults to -1.0.
        args: Variable length argument list passed to the base class.
        kwargs: Arbitrary keyword arguments passed to the base class.
    """

    def __init__(
        self,
        sizex: int,
        gamma: float = 0.0,
        scale: float = 2.0,
        shift: float = -1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.sizex = sizex
        self.gamma = gamma
        self.scale = scale
        self.shift = shift

    def gen_data(self, d: int, Sb: int = 10, Sr: int = 10) -> Dict[str, Tensor]:
        """
        Generate the data.

        Args:
            d (int): The number of inputs to shatter.
            Sb (int): The number of binary samples to check shattering.
                Defaults to 10.
            Sr (int): The number of level offset samples to check shattering.
                Defaults to 10.

        Returns:
            data (Dict[str, Tensor]): A dictionary containing the generated data.
        """

        X = gen_synthetic_features(
            d, self.sizex, self.seed, self.scale, self.shift, self.device
        )
        b = generate_samples_b_fat(d, Sb)
        r = generate_samples_r_fat(d, Sr)
        Y = gen_synthetic_labels_fat(b, r, self.gamma, self.device)

        data = {"X": X, "Y": Y, "b": b, "r": r}

        return data


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
    Generates a dataset of inputs x and outputs y for a QNN.

    Args:
        N (int): The number of inputs for the QNN.
        sizex (int): The size of each input.
        num_samples (int): The number of output samples to generate. Defaults to 10.
        seed (int, optional): The random seed to use for generating the dataset.
            Defaults to None.
        scale (float, optional): The re-scaling factor for uniform random numbers
            in [0,1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform random numbers [0,1].
            Defaults to -1.0.
        device (Device, optional): Torch device to run on. Defaults to CPU.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the input
            tensor x of shape (N, sizex) and the output
        tensor y of shape (num_samples, N).
    """

    scale = 2.0
    shift = -1.0

    if seed is not None:
        torch.manual_seed(seed)
    x = (
        scale
        * torch.rand(N, sizex, dtype=torch.float64, device=device, requires_grad=False)
        + shift
    )
    y = (
        scale
        * torch.rand(
            num_samples, N, dtype=torch.float64, device=device, requires_grad=False
        )
        + shift
    )

    return x, y


def generate_samples_b_fat(d: int, S: int) -> np.ndarray:
    """
    Generate S unique samples of b from {0, 1}^d.

    Args:
        d (int): Number of input data samples for shattering.
        S (int): Number of binary samples to check shattering
            (for scalability if d is large).

    Returns:
        numpy.ndarray: An array of shape (S, d) containing unique samples of b.
    """

    max_values = 2**d
    if S >= max_values:
        possible_values = list(product([0, 1], repeat=d))
        return np.array(possible_values)
    else:
        samples: Set[Tuple[int, ...]] = set()
        while len(samples) < S:
            b_sample = tuple(np.random.randint(0, 2, size=d))
            samples.add(b_sample)
        return np.array(list(samples))


def generate_samples_r_fat(d: int, S: int) -> np.ndarray:
    """
    Generate S samples of r from [0, 1]^d using Latin Hypercube Sampling.

    Args:
        d (int): Dimension of the feature space.
        S (int): Number of samples to generate.

    Returns:
        numpy.ndarray: An array of shape (S, d) containing samples of r.
    """

    sampler = qmc.LatinHypercube(d=d)
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

    Args:
        d (int): The number of inputs x to generate.
        sizex (int): The size of each input.
        seed (int, optional): The random seed to use for generating the features.
            Defaults to None.
        scale (float, optional): The re-scaling factor for uniform random numbers
            in [0,1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform random numbers [0,1].
            Defaults to -1.0.
        device (Device, optional): Torch device to run on. Defaults to CPU.

    Returns:
        Tensor: Tensor X of shape (d, sizex).
    """

    scale = 2.0
    shift = -1.0

    if seed is not None:
        torch.manual_seed(seed)

    x = (
        scale
        * torch.rand(d, sizex, dtype=torch.float64, device=device, requires_grad=False)
        + shift
    )

    return x


def gen_synthetic_labels_fat(
    b: np.ndarray,
    r: np.ndarray,
    gamma: float = 0.0,
    device: Device = torch.device("cpu"),
) -> Tensor:
    """
    Generate constant label values equal to r_i + gamma
    when b_i = 1 and r_i - gamma when b_i = 0.

    Args:
        b (numpy.ndarray): An array of shape (Sb,d)
            containing Sb samples of d-dim binary values.
        r (numpy.ndarray): An array of shape (Sr,d)
            containing Sr samples of d-dim real values in [0, 1].
        gamma (float, optional): The fat-shattering margin value.
            Defaults to 0.0.
        device (Device, optional): Torch device to run on. Defaults to CPU.

    Returns:
        Tensor: Y of shape (Sr, Sb, d).

    Raises:
        ValueError: If the length of b[0] is not the same as the length of r[0].
    """

    Sb = len(b)
    Sr = len(r)
    d = d1 = len(b[0])
    d2 = len(r[0])

    if d1 != d2:
        raise ValueError(
            f"The length of b[0] and r[0] are {d1} and {d2}. "
            f"Should be constant and the same."
        )

    labels = np.zeros((Sr, Sb, d))
    for sr in range(Sr):
        for sb in range(Sb):
            for i in range(d):
                if b[sb][i] == 1:
                    labels[sr, sb, i] = r[sr, i] + gamma
                else:
                    labels[sr, sb, i] = r[sr, i] - gamma

    y = torch.tensor(labels, dtype=torch.float64, device=device, requires_grad=False)

    return y

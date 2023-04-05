from typing import Optional, TypeAlias, TypeVar, Generic, Tuple, Dict
from abc import ABC, abstractmethod
import torch

Tensor: TypeAlias = torch.Tensor
Device: TypeAlias = torch.device
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

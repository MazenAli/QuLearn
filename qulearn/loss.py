from typing import Optional

import torch

from .types import Loss, Tensor


class RademacherLoss(Loss):
    """
    Computes the loss function required to estimate emprical Rademacher complexity.

    :param sigmas: 1D Tensor of sigmas.
    :type sigmas: Tensor
    """

    def __init__(self, sigmas: Tensor) -> None:
        super().__init__()

        self._check_sigmas(sigmas)
        self.sigmas = sigmas

    def forward(self, output: Tensor, _: Optional[Tensor] = None) -> Tensor:
        """
        Compute loss = -1/m * sum_k sigma_k*f_k.

        :param output: Predicted value tensor.
        :type output: Tensor
        :param _: Ignored. Defaults to None.
        :type _: Tensor
        :return: Loss value.
        :rtype: Tensor
        :raises ValueError: If output not a 1D Tensor, or not the same length as sigmas.
        """

        len_shape = len(output.shape)
        if len_shape != 2:
            raise ValueError(f"output (dim = {len_shape}) should be a 2D tensor")
        second_dim = output.shape[1]
        if second_dim != 1:
            raise ValueError(f"output second dim should be 1 not {second_dim}")

        len_out = output.shape[0]
        len_sigmas = self.sigmas.shape[0]
        if len_out != len_sigmas:
            raise ValueError(
                f"output length ({len_out}) does not match sigmas length ({len_sigmas})"
            )

        loss = -(self.sigmas * output[:, 0]).mean()

        return loss

    def _check_sigmas(self, sigmas: Tensor) -> None:
        """
        Ensures sigmas has the expected format.

        :param sigmas: sigmas Tensor.
        :type sigmas: Tensor
        :return: None
        :raises ValueError: If sigmas has invalid shape or values.
        """

        len_shape = len(sigmas.shape)

        if len_shape != 1:
            raise ValueError(f"sigmas (dim={len_shape}) should be a 1D tensor")

        if not torch.all((sigmas == 1) | (sigmas == -1)):
            raise ValueError("All sigmas must be 1 or -1")

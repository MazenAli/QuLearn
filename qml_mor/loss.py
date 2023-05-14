from typing import Optional

# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
Loss: TypeAlias = torch.nn.Module


class RademacherLoss(Loss):
    """
    Computes the loss function required to estimate emprical Rademacher complexity.

    Args:
        sigmas (Tensor): 1D Tensor of sigmas.
    """

    def __init__(self, sigmas: Tensor) -> None:
        super().__init__()

        self._check_sigmas(sigmas)
        self.sigmas = sigmas

    def forward(self, output: Tensor, _: Optional[Tensor] = None) -> Tensor:
        """
        Compute loss = -1/m * sum_k sigma_k*f_k.

        Args:
            output (Tensor): Predicted value tensor.
            _ (Tensor, optional): Ignored. Defaults to None.

        Returns:
            Tensor: Loss value.

        Raises:
            ValueError: If output not a 1D Tensor, or not the same length as sigmas.
        """

        len_shape = len(output.shape)
        if len_shape != 1:
            raise ValueError(f"output (dim = {len_shape}) should be a 1D tensor")

        len_out = output.shape[0]
        len_sigmas = self.sigmas.shape[0]
        if len_out != len_sigmas:
            raise ValueError(
                f"output length ({len_out}) does not match sigmas length ({len_sigmas})"
            )

        loss = -(self.sigmas * output).mean()

        return loss

    def _check_sigmas(self, sigmas: Tensor) -> None:
        """
        Ensures sigmas has the expected format.

        Args:
            sigmas (Tensor): sigmas Tensor.

        Returns:
            None

        Raises:
            ValueError: If sigmas has invalid shape or values.
        """

        len_shape = len(sigmas.shape)

        if len_shape != 1:
            raise ValueError(f"sigmas (dim={len_shape}) should be a 1D tensor")

        if not torch.all((sigmas == 1) | (sigmas == -1)):
            raise ValueError("All sigmas must be 1 or -1")

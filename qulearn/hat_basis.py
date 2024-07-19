# for python < 3.10
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

from typing import Tuple

import torch

Tensor: TypeAlias = torch.Tensor


class HatBasis:
    """
    Linear 1D hat basis for the range [a, b] and a specified number of nodes,
    including a and b.

    :param a: Left end point.
    :type a: float
    :param b: Right end point.
    :type b: float
    :param num_nodes: Number of nodes.
    :type num_nodes: int
    """

    def __init__(self, a: float, b: float, num_nodes: int) -> None:
        self.a = a
        self.b = b
        self.num_nodes = num_nodes
        self.num_segments = num_nodes - 1
        self.segment_length = (b - a) / self.num_segments

    def position(self, x: Tensor) -> Tensor:
        """
        Find the index of the grid point left of x.

        :param x: A tensor containing the values for which the position indexes are to be found.
        :type x: Tensor
        :returns: A tensors of position indexes.
        The position indices are -1 for values left of `a`, and -2 for values right of `b`.
        :rtype: Tensor
        """

        left_of_a = x < self.a
        right_of_b = x > self.b

        within_range = torch.logical_not(torch.logical_or(left_of_a, right_of_b))
        position = torch.zeros_like(x)
        position[within_range] = ((x[within_range] - self.a) / self.segment_length).floor()

        position[left_of_a] = -1
        position[right_of_b] = -2

        return position

    def grid_points(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Finds the grid points surrounding given values in the discretized space.

        :param x: A tensor containing the values for which the surrounding
        grid points are to be found.
        :type x: Tensor
        :returns: A tuple of two tensors. The first tensor contains the left boundary points
        of the segments, the second tensor contains the right boundary points of the segments.
        :rtype: Tuple[Tensor, Tensor]
        """

        left_point = torch.zeros_like(x)
        right_point = torch.zeros_like(x)

        left_of_a = x < self.a
        left_point[left_of_a] = self.a - self.segment_length
        right_point[left_of_a] = self.a

        right_of_b = x > self.b
        left_point[right_of_b] = self.b
        right_point[right_of_b] = self.b + self.segment_length

        within_range = torch.logical_not(torch.logical_or(left_of_a, right_of_b))

        position = self.position(x)

        left_point[within_range] = self.a + position[within_range] * self.segment_length
        right_point[within_range] = left_point[within_range] + self.segment_length

        return left_point, right_point

    def nonz_vals(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Output the 2 nonzero values for any given x.
        Finds the grid points surrounding given values in the discretized space.

        :param x: Input variables.
        :type x: Tensor
        :returns: A tuple of 2 nonzero values.
        :rtype: Tuple[Tensor, Tensor]
        """

        left, right = self.grid_points(x)
        first = (right - x) / self.segment_length
        second = (x - left) / self.segment_length
        return first, second

    def eval_basis_vector(self, x: Tensor) -> Tensor:
        """
        Evaluate basis vector at points x.

        :param x: Input variables.
        :type x: Tensor
        :returns: Vector of all basis functions evaluated at x.
        :rtype: Tuple[Tensor, Tensor]
        """

        nodes = torch.linspace(-1, 1, self.num_nodes, dtype=x.dtype)
        values = torch.zeros(x.shape[0], self.num_nodes, dtype=x.dtype)
        for i in range(1, self.num_nodes):
            mask = (nodes[i - 1] <= x) & (x <= nodes[i])
            values[mask, i - 1] = (nodes[i] - x[mask]) / self.segment_length
            values[mask, i] = (x[mask] - nodes[i - 1]) / self.segment_length

        return values.squeeze(0)

def estimate_fat_shattering_dimension(d, hidden_units, max_samples, step_size, gamma):
    """
    Estimate the fat-shattering dimension for a model with a given architecture.

    Parameters
    ----------
    d : int
        Dimension of the feature space.
    hidden_units : int
        Number of hidden units in the model.
    max_samples : int
        Maximum number of samples to use for fat-shattering dimension estimation.
    step_size : int
        Step size for incrementing the number of samples during estimation.
    gamma : float
        The margin value.

    Returns
    -------
    int
        The estimated fat-shattering dimension.
    """


def check_shattering(model, X, r, gamma):
    """
    Check if the model shatters the given samples X with margin gamma.

    Parameters
    ----------
    model : Keras model
        The trained model.
    X : numpy.ndarray
        An array of shape (n_samples, d) containing input samples.
    r : numpy.ndarray
        An array of shape (n_samples,) containing r values.
    gamma : float
        The margin value.

    Returns
    -------
    bool
        True if the model shatters the samples, False otherwise.
    """
    # Function implementation


def generate_samples_b(d, S):
    """
    Generate S unique samples of b from {0, 1}^d.

    Parameters
    ----------
    d : int
        Dimension of the feature space.
    S : int
        Number of samples to generate.

    Returns
    -------
    numpy.ndarray
        An array of shape (S, d) containing unique samples of b.
    """
    # Function implementation


def generate_samples_r(d, S):
    """
    Generate S samples of r from [0, 1]^d using Latin Hypercube Sampling.

    Parameters
    ----------
    d : int
        Dimension of the feature space.
    S : int
        Number of samples to generate.

    Returns
    -------
    numpy.ndarray
        An array of shape (S, d) containing samples of r.
    """
    # Function implementation


def gen_synthetic_features(
    d: int,
    sizex: int,
    seed: Optional[int] = None,
    scale: float = 2.0,
    shift: float = -1.0,
    cuda: bool = False,
) -> Tensor:
    """
    Generates d inputs x  of dimension sizex sampled uniformly from scale*[0,1]+shift.

    Args:
        d (int): The number of inputs x to generate.
        sizex (int): The size of each input.
        seed (int, optional): The random seed to use for generating the features.
            Defaults to None.
        scale (float, optional): The re-scaling factor for uniform random numbers
            in [0,1]. Defaults to 2.0.
        shift (float, optional): The shift value for uniform random numbers [0,1].
            Defaults to -1.0.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        Tensor: Tensor X of shape (d, sizex).
    """

    scale = 2.0
    shift = -1.0

    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    if seed is not None:
        torch.manual_seed(seed)

    x = (
        scale
        * torch.rand(N, sizex, dtype=torch.float64, device=device, requires_grad=False)
        + shift
    )

    return x


def gen_synthetic_labels(
    b: np.ndarray,
    r: np.ndarray,
    gamma: float = 0.0,
    c: float = 0.0,
    cuda: bool = False,
) -> Tensor:
    """
    Generate a constant labels values equal to r_i + gamma + c
    when b_i = 1 and r_i - gamma - c when b_i = 0.

    Args:
        b (numpy.ndarray): An array of shape (Sb,d)
            containing Sb samples of d-dim binary values.
        r (numpy.ndarray): An array of shape (Sr,d)
            containing Sr samples of d-dim real values in [0, 1].
        gamma (float, optional): The fat-shattering margin value.
            Defaults to 0.0.
        c (float, optional): The constant value added to the margin.
            Defaults to 0.0.
        cuda (bool, optional): Set True if run on GPU. Defaults to False.

    Returns:
        Tensor: Y of shape (Sr, Sb, d).

    Raises:
            ValueError: If the length of b[0] is not the same as the length of r[0].
    """

    Sb = len(b)
    Sr = len(r)
    d = d1 = len(b[0])
    d2 = len(r[[0]])

    if d1 != d2:
        raise ValueError(
            f"The length of b[0] and r[0] are {d1} and {d2}. Should be constant and the same."
        )

    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda")

    labels = np.zeros(Sr, Sb, d)
    for sr in range(Sr):
        for sb in range(Sb):
            for i in range(d):

                if b[sb][i] == 1:
                    labels[sr, sb, i] = r[sr, i] + (gamma + c)
                else:
                    labels[sr, sb, i] = r[sr, i] - (gamma + c)

    y = torch.tensor(labels, dtype=torch.float64, device=device, requires_grad=False)

    return y


def normalize_const(weights: Tensor, gamma: float) -> float:
    """
    Compute a normalization constant given a tensor of weights
    and the margin parameter gamma.

    Args:
        weights (Tensor): Tensor of weights
        gamma (float): Margin parameter.

    Returns:
        float: A positive real-valued normalization constant.
    """

    V = torch.norm(weights, p=1)
    C = (V / gamma) ** 2 * np.log2(V / gamma)

    return C

from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLogistic(AttributesAbstract):
    """
    AttributesLogistic class contains the common attributes & methods to update the LogisticModel's attributes.

    Attributes
    ----------
    dimension: `int`
    source_dimension: `int`
    betas: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
        positions = exp(realizations['g']) such that p0 = 1 / (1+exp(g))
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    velocities: `torch.Tensor` (default None)
    name: `str` (default 'logistic')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: `tuple` [`str`] (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``positions``, ``deltas`` & ``mixing_matrix``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self, dimension, source_dimension):
        """
        Instantiate a AttributesLogistic class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        super().__init__(dimension, source_dimension)
        self.name = 'logistic'
        if (type(dimension) != int) & (type(source_dimension) != int):
            raise ValueError("For AttributesLogistic you must provide integer io for the parameters"
                             " `dimension` and `source_dimension`!")

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to velocities v0 for the inner product
        implied by the metric. It is equivalent to be a base orthogonal to v0 / (p0^2 (1-p0)^2 for the euclidean norm.
        """
        if self.source_dimension == 0:
            return

        # Compute regularizer to work in the euclidean space
        metric_normalization = (1 + self.positions).pow(4) / self.positions.pow(2)
        dgamma_t0 = self.velocities * metric_normalization

        # Compute Q
        self._compute_Q(dgamma_t0)

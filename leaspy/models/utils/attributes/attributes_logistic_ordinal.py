from .attributes_logistic import AttributesLogistic
import torch


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLogisticOrdinal(AttributesLogistic):
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

    def __init__(self, dimension, source_dimension, ordinal_infos):
        """
        Instantiate a AttributesLogistic class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        super().__init__(dimension, source_dimension)
        self.ordinal_infos = ordinal_infos
        self.deltas = None
#        self.max_level = max([feat["nb_levels"] for feat in ordinal_infos])
        self.update_possibilities = ('all', 'g', 'v0', 'betas','deltas')

    def get_attributes(self):
        """
        Returns the attributes:
        ``positions`` modified by ``positions_deltas``,
        ``velocities`` modified by ``velocities_deltas``
        & ``mixing_matrix``.

        Returns
        -------
        - positions: `torch.Tensor`
        - velocities: `torch.Tensor`
        - mixing_matrix: `torch.Tensor`
        """
        return self.positions, self.velocities, self.mixing_matrix

    def get_deltas(self):

        return self.deltas

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]
            Must be one of - "all", "g", "v0", "betas". Raise an error otherwise.
            "g" correspond to the attribute ``positions``.
            "v0" correspond to the attribute ``velocities``.
        values: `dict` [`str`, `torch.Tensor`]
            New values used to update the model's group average parameters
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_deltas = False
        compute_positions = False
        compute_velocities = False

        if 'all' in names_of_changed_values:
            names_of_changed_values = self.update_possibilities  # make all possible updates

        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'deltas' in names_of_changed_values:
            compute_deltas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('v0' in names_of_changed_values) or ('xi_mean' in names_of_changed_values):
            compute_velocities = True

        if compute_betas:
            self._compute_betas(values)
        if compute_deltas:
            self._compute_deltas(values)
        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)


        # TODO : Check if the condition is enough
        if (compute_positions or compute_velocities) and (self.name != 'univariate'):
            self._compute_orthonormal_basis()
        if (compute_positions or compute_velocities or compute_betas) and (self.name != 'univariate'):
            self._compute_mixing_matrix()

    def _compute_deltas(self, values):

        """
                Update the attribute ``deltas``.

                Parameters
                ----------
                values: `dict` [`str`, `torch.Tensor`]
                """
        self.deltas = torch.exp(values['deltas'])

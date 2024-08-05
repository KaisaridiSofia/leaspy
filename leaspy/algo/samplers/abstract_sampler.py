import torch


class AbstractSampler:
    """
    Abstract sampler class.

    Attributes
    ----------
    acceptation_temp : :class:`torch.Tensor`
        Acceptation rate for the sampler in MCMC-SAEM algorithm
        Keep the history of the last `temp_length` last steps
    name: str
        Name of variable
    shape: tuple
        Shape of variable
    temp_length: int
        Deepness of the history kept in the acceptation rate `acceptation_temp`
        Length of the `acceptation_temp` torch tensor
    """

    def __init__(self, info, n_patients):
        self.acceptation_temp = None
        self.name = info["name"]
        self.shape = info["shape"]
        self.temp_length = 25  # For now the same between pop and ind #TODO this is an hyperparameter
        if info["type"] == "population":
            self.type = 'pop'
            # Initialize the acceptation history
            if len(self.shape) < 2:
                self.acceptation_temp = torch.zeros(size=self.shape).repeat(self.temp_length,
                                                                            1)  # convention : shape of pop is 2D
            elif len(self.shape) == 2:
                self.acceptation_temp = torch.zeros(size=self.shape).repeat(self.temp_length, 1, 1)
            else:
                raise ValueError("Dimension of population variable >2")
        elif info["type"] == "individual":
            self.type = 'ind'
            # Initialize the acceptation history
            self.acceptation_temp = torch.zeros(size=(n_patients,)).repeat(self.temp_length, 1)

    def _group_metropolis_step(self, alpha):
        """
        Compute the acceptance decision (0. for False & 1. for True).

        Parameters
        ----------
        alpha : :class:`torch.Tensor`

        Returns
        -------
        accepted : :class:`torch.Tensor`
            Acceptance decision (0. or 1.). The logs must be one dimensional (i.e. accepted.ndim = 1)
        """
        accepted = (torch.rand(alpha.size(0)) < alpha).float()  # TODO: change for boolean?
        return accepted

    def _metropolis_step(self, alpha):
        """
        Compute the Metropolis acceptance decision
        If better (alpha>=1) : accept
        If worse (alpha<1) : accept with probability alpha

        Parameters
        ----------
        alpha : :class:`torch.Tensor`

        Returns
        -------
        int
            acceptance decision (0 or 1)
        """

        accepted = 0
        if alpha >= 1:
            # Case 1: we improved the LogL
            accepted = 1
        else:
            # Case 2: we decreased the LogL
            # Sample a realization from uniform law
            realization = torch.rand(1)
            # Choose to keep a lesser parameter value from it
            if realization < alpha:
                accepted = 1
        return accepted  # TODO: change for boolean?

    def _update_acceptation_rate(self, accepted):
        """
        Update acceptation rate from history of boolean accepted values for each dimension of each variable (except sources)

        Parameters
        ----------
        accepted : :class:`torch.Tensor`
        """

        # Ad the new acceptation result
        if self.type == "pop":
            self.acceptation_temp = torch.cat(
                [self.acceptation_temp, accepted.reshape(self.acceptation_temp.shape[1:]).unsqueeze(0)])
        elif self.type == "ind":
            self.acceptation_temp = torch.cat(
                [self.acceptation_temp, accepted.unsqueeze(0)])
        else:
            raise ValueError("Nor pop or ind")

        self.acceptation_temp = self.acceptation_temp[1:]

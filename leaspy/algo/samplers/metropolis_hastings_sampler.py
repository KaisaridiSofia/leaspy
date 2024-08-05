#import itertools

import torch

from .abstract_sampler import AbstractSampler


class MetropolisHastingsSampler(AbstractSampler):
    """
    MH sampler class.

    Parameters
    ----------
    info: dict
        Informations on variable to be sampled
    n_patients: int > 0
        Number of individual (used for variable with ``info['type'] == 'individual'``)
    """

    def __init__(self, info, n_patients):
        super().__init__(info, n_patients)

        self.std = None

        if info["type"] == "population":
            # Proposition variance is the same for all dimensions
            self.std = 0.005 * torch.ones(self.shape) # TODO hyperparameter here
            self.acceptation_temp = torch.zeros(self.temp_length,1)
        elif info["type"] == "individual":
            # Proposition variance is adapted independantly on each patient, but is the same for multiple dimensions
            # TODO : gérer les shapes !!! Necessary for sources
            self.std = torch.tensor([0.1] * n_patients * int(self.shape[0]),
                                    dtype=torch.float32).reshape(n_patients,int(self.shape[0]))
        else:
            raise NotImplementedError

        # Acceptation rate
        self.counter_acceptation = 0

        self.previous_attachment = None
        self.previous_regularity = None

    def sample(self, data, model, realizations, temperature_inv, previous_attachment=None, clusters=None, temper="regularization"):
        """
        Sample either as population or individual.

        Modifies in-place the realizations object.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        temperature_inv : float > 0
        """
        # TODO is data / model / realizations supposed to be in sampler ????

        if self.type == 'pop':
            return self._sample_population_realizations(data, model, realizations, temperature_inv,
                                                        previous_attachment=previous_attachment,
                                                        clusters=clusters, temper=temper)
        else:
            return self._sample_individual_realizations(data, model, realizations, temperature_inv,
                                                        previous_attachment=previous_attachment, temper=temper)

    def _proposal(self, val, std):
        """
        Proposal value around the current value with sampler standard deviation.

        Parameters
        ----------
        val

        Returns
        -------
        value around `val`
        """
        # return val+self.distribution.sample(sample_shape=val.shape)
        # Torch distribution
        distribution = torch.distributions.normal.Normal(loc=0.0, scale=std)
        return val + distribution.sample()

    def _update_std(self):
        """
        Update standard deviation of sampler according to current frequency of acceptation.

        Adaptative std is known to improve sampling performances.
        Std is increased if frequency of acceptation > 40%, and decreased if <20%, so as
        to stay close to 30%.
        """

        self.counter_acceptation += 1

        if self.counter_acceptation == self.temp_length:
            mean_acceptation = self.acceptation_temp.mean(0)

            if self.type == 'pop':

                if mean_acceptation < 0.2:
                    self.std *= 0.9

                elif mean_acceptation > 0.4:
                    self.std *= 1.1

            else:

                idx_toolow = mean_acceptation < 0.2
                idx_toohigh = mean_acceptation > 0.4

                self.std[idx_toolow] *= 0.9
                self.std[idx_toohigh] *= 1.1

            # reset acceptation temp list
            self.counter_acceptation = 0

    def _set_std(self, std):
        self.std = std

    def _sample_population_realizations(self, data, model, realizations, temperature_inv, previous_attachment=None,
                                        clusters=None, temper="regularization"):
        """
        For each dimension (1D or 2D) of the population variable, compute current attachment and regularity.
        Propose a new value for the given dimension of the given population variable,
        and compute new attachment and regularity.
        Do a MH step, keeping if better, or if worse with a probability.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        temperature_inv : float > 0
        """

        if clusters is None:  # to keep computations fluid we set clusters to neutral scalar
            clusters = 1.

        realization = realizations[self.name]

        self.previous_attachment = previous_attachment

#        index = [e for e in itertools.product(*[range(s) for s in shape_current_variable])]
        # Compute the attachment and regularity
        # previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
        # previous_regularity = model.compute_regularity_realization(realization).sum()
        if self.previous_attachment is None:
            self.previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        if self.previous_regularity is None:
            self.previous_regularity = model.compute_regularity_realization(realization).sum()

        # Keep previous realizations and sample new ones
        previous_reals_pop = realization.tensor_realizations.clone()
        new_val = self._proposal(realization.tensor_realizations, self.std)
        realization.tensor_realizations = new_val

        # Update intermediary model variables if necessary
        model.update_MCMC_toolbox([self.name], realizations)

        # Compute the attachment and regularity
        new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        new_regularity = model.compute_regularity_realization(realization).sum()

        if temper == "regularization":
            alpha = torch.exp(-((new_regularity - self.previous_regularity) * temperature_inv +
                                (clusters * (new_attachment - self.previous_attachment)).sum()))
        elif temper == "all":
            alpha = torch.exp(-((new_regularity - self.previous_regularity) +
                                (clusters * (new_attachment - self.previous_attachment)).sum()) * temperature_inv)
        else:
            alpha = torch.exp(-((new_regularity - self.previous_regularity) +
                                (clusters * (new_attachment - self.previous_attachment)).sum()))
        accepted = self._metropolis_step(alpha)

        # Revert if not accepted
        if not accepted:
            # Revert realizations
            realization.tensor_realizations = previous_reals_pop
            # Update intermediary model variables if necessary
            model.update_MCMC_toolbox([self.name], realizations)
            # force re-compute on next iteration -> No need...
#            self.previous_attachment = self.previous_regularity = None
        else:
            self.previous_attachment = new_attachment
            self.previous_regularity = new_regularity

        self._update_acceptation_rate(torch.tensor([[accepted]], dtype=torch.float32))
        self._update_std()

        #current_attachment = self.previous_attachment
        # Reset previous attachment and regularity !!!
        #self.previous_attachment = self.previous_regularity = None

        return self.previous_attachment

    def _sample_individual_realizations(self, data, model, realizations, temperature_inv, previous_attachment=None, temper="regularization"):
        """
        For each indivual variable, compute current patient-batched attachment and regularity.
        Propose a new value for the individual variable,
        and compute new patient-batched attachment and regularity.
        Do a MH step, keeping if better, or if worse with a probability.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        temperature_inv : float > 0
        """

        # Compute the attachment and regularity
        realization = realizations[self.name]

        if previous_attachment is None:
            previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        # use realizations => use all individual parameters to compare reconstructions vs values
        # previous_attachment.ndim = 1
        previous_regularity = model.compute_regularity_realization(realization).sum(dim=1).reshape(data.n_individuals)
        # compute log-likelihood of just the given parameter (tau or xi or ...)

        # Keep previous realizations and sample new ones
        previous_reals = realization.tensor_realizations.clone()
        realization.tensor_realizations = self._proposal(realization.tensor_realizations, self.std)
        # Add perturbations to previous observations

        # Compute the attachment and regularity
        new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations)
        new_regularity = model.compute_regularity_realization(realization).sum(dim=1).reshape(data.n_individuals)

        if temper == "regularization":
            alpha = torch.exp(-((new_regularity - previous_regularity) * temperature_inv +
                                (new_attachment - previous_attachment)))
        elif temper == "all":
            alpha = torch.exp(-((new_regularity - previous_regularity) +
                                (new_attachment - previous_attachment)) * temperature_inv)
        else:
            alpha = torch.exp(-((new_regularity - previous_regularity) +
                                (new_attachment - previous_attachment)))

        accepted = self._group_metropolis_step(alpha)  # accepted.ndim = 1
        self._update_acceptation_rate(accepted)
        self._update_std()
        ##### PEUT ETRE PB DE SHAPE
        accepted_ = accepted.unsqueeze(1)
        realization.tensor_realizations = accepted_ * realization.tensor_realizations + (1. - accepted_) * previous_reals

        return accepted * new_attachment + (1. - accepted) * previous_attachment

#import itertools

import torch

from .abstract_sampler import AbstractSampler


class RMALA(AbstractSampler):
    """
    Riemannian MALA sampler class.

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
            self.std = 0.001 * torch.ones(self.shape) # TODO hyperparameter here
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
        self.moments = None

    def sample(self, data, model, realizations, temperature_inv, moments=None, clusters=None, temper="regularization"):
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
                                                        moments=moments,
                                                        clusters=clusters, temper=temper)
        else:
            return self._sample_individual_realizations(data, model, realizations, temperature_inv,
                                                        moments=moments, temper=temper)

    def _proposal(self, val, std, derivative, metric):
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
        sample = distribution.sample()
        eps = 10e-4
        if self.type == "pop":
            sample = sample.reshape(-1)
            metric_reg = eps * torch.eye(metric.shape[-1])
        else:
            sample = sample.reshape(self.std.shape[0], -1, 1)
            derivative = derivative.unsqueeze(-1)
            metric_reg = eps * torch.eye(metric.shape[-1]).expand(self.std.shape[0], -1, -1)
        L, Q = torch.linalg.eigh(metric + metric_reg)
        Q_inv = Q.transpose(-1, -2)
        metric_inv = (Q * (1. / L.unsqueeze(1))).matmul(Q_inv)
        sqrt_metric_inv = (Q * (1. / torch.sqrt(L).unsqueeze(1))).matmul(Q_inv)
        out = val + sqrt_metric_inv.matmul(sample).reshape(self.std.shape) + (std * std) * metric_inv.matmul(
            derivative).reshape(self.std.shape) / 2

        return out

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

    def _sample_population_realizations(self, data, model, realizations, temperature_inv, moments=None,
                                        clusters=None, temper="regularization", **kwargs):
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

        self.moments = moments

#        index = [e for e in itertools.product(*[range(s) for s in shape_current_variable])]
        # Compute the attachment and regularity
        # previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations).sum()
        # previous_regularity = model.compute_regularity_realization(realization).sum()
        if self.moments is None or not self.name in self.moments["attachment"][1]:
            self.moments = {}
            individual_parameters = model.get_param_from_real(realizations)
            self.moments["attachment"] = model.compute_moments_model(data.timepoints, individual_parameters, order=2,
                                                                     selected_variables = [self.name],
                                                                     attribute_type='MCMC')
            self.moments["regularization"] = model.compute_moments_regularization(realizations, order=2,
                                                                                  selected_variables=[self.name],
                                                                                  attribute_type='MCMC')

        # Keep previous realizations and sample new ones
        previous_reals_pop = realization.tensor_realizations.clone()
        derivative = model.compute_loss_derivative(data, realization.tensor_realizations,
                                                   self.moments["attachment"][0],
                                                   self.moments["attachment"][1],
                                                   self.moments["regularization"][1],
                                                   sum_individuals=True)
        metric = self.moments["attachment"][2].nansum(dim=(0, 1, 2)) + self.moments["regularization"][2]
        new_val = self._proposal(realization.tensor_realizations, self.std, -derivative[self.name], metric)
        realization.tensor_realizations = new_val

        # Update intermediary model variables if necessary
        model.update_MCMC_toolbox([self.name], realizations)

        # Compute the attachment and regularity
        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations,
                                                                                  model_values=self.moments["attachment"][0])
        individual_parameters = model.get_param_from_real(realizations)
        new_values = model.compute_moments_model(data.timepoints, individual_parameters, order=0,
                                                     selected_variables=[self.name],
                                                     attribute_type='MCMC')
        new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations,
                                                            model_values=new_values)
        new_regularity = model.compute_moments_regularization(realizations, order=0,
                                                              selected_variables=[self.name],
                                                              attribute_type='MCMC')[self.name]

        if temper == "regularization":
            alpha = torch.exp(-((new_regularity - self.moments["regularization"][0][self.name]).sum() * temperature_inv +
                                (clusters * (new_attachment - previous_attachment)).sum()))
        elif temper == "all":
            alpha = torch.exp(-((new_regularity - self.moments["regularization"][0][self.name]).sum() +
                                (clusters * (new_attachment - previous_attachment)).sum()) * temperature_inv)
        else:
            alpha = torch.exp(-((new_regularity - self.moments["regularization"][0][self.name]).sum() +
                                (clusters * (new_attachment - previous_attachment)).sum()))
        accepted = self._metropolis_step(alpha)

        # Revert if not accepted
        if not accepted:
            # Revert realizations
            realization.tensor_realizations = previous_reals_pop
            # Update intermediary model variables if necessary
            model.update_MCMC_toolbox([self.name], realizations)

        self._update_acceptation_rate(torch.tensor([[accepted]], dtype=torch.float32))
        self._update_std()

        return None

    def _sample_individual_realizations(self, data, model, realizations, temperature_inv, moments=None,
                                        temper="regularization", **kwargs):
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
        self.moments = moments

        if self.moments is None or not self.name in self.moments["attachment"][1]:
            self.moments = {}
            individual_parameters = model.get_param_from_real(realizations)
            self.moments["attachment"] = model.compute_moments_model(data.timepoints, individual_parameters, order=2,
                                                                     selected_variables=[self.name],
                                                                     attribute_type='MCMC')
            self.moments["regularization"] = model.compute_moments_regularization(realizations, order=2,
                                                                                  selected_variables=[self.name],
                                                                                  attribute_type='MCMC')
        # compute log-likelihood of just the given parameter (tau or xi or ...)

        # Keep previous realizations and sample new ones
        previous_reals = realization.tensor_realizations.clone()
        derivative = model.compute_loss_derivative(data, realization.tensor_realizations,
                                                   self.moments["attachment"][0],
                                                   self.moments["attachment"][1],
                                                   self.moments["regularization"][1],
                                                   sum_individuals=False)
        metric = self.moments["attachment"][2].nansum(dim=(1, 2)) + self.moments["regularization"][2]
        new_val = self._proposal(realization.tensor_realizations, self.std, -derivative[self.name], metric)
        realization.tensor_realizations = new_val
        # Add perturbations to previous observations

        # Compute the attachment and regularity
        previous_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations,
                                                                                  model_values=
                                                                                  self.moments["attachment"][0])
        individual_parameters = model.get_param_from_real(realizations)
        new_values = model.compute_moments_model(data.timepoints, individual_parameters, order=0,
                                                 selected_variables=[self.name],
                                                 attribute_type='MCMC')
        new_attachment = model.compute_individual_attachment_tensorized_mcmc(data, realizations,
                                                                             model_values=new_values)
        new_regularity = model.compute_moments_regularization(realizations, order=0,
                                                              selected_variables=[self.name],
                                                              attribute_type='MCMC')[self.name]

        if temper == "regularization":
            alpha = torch.exp(-((new_regularity - self.moments["regularization"][0][self.name]).sum() * temperature_inv +
                                (new_attachment - previous_attachment)))
        elif temper == "all":
            alpha = torch.exp(-((new_regularity - self.moments["regularization"][0][self.name]).sum() +
                                (new_attachment - previous_attachment)) * temperature_inv)
        else:
            alpha = torch.exp(-((new_regularity - self.moments["regularization"][0][self.name]).sum() +
                                (new_attachment - previous_attachment)))

        accepted = self._group_metropolis_step(alpha)  # accepted.ndim = 1
        self._update_acceptation_rate(accepted)
        self._update_std()
        ##### PEUT ETRE PB DE SHAPE
        accepted_ = accepted.unsqueeze(1)
        realization.tensor_realizations = accepted_ * realization.tensor_realizations + (1. - accepted_) * previous_reals

        return None

import torch
import math

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_factory import AttributesFactory


class MultivariateIPMixtureModel(AbstractMultivariateModel):
    def __init__(self, name, nb_clusters=1):
        super(MultivariateIPMixtureModel, self).__init__(name)
        self.parameters["v0"] = None
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef
        self.nb_clusters = nb_clusters
        for k in range(self.nb_clusters):
            self.parameters[f'tau_xi_{k}_mean'] = None
            self.parameters[f'tau_xi_{k}_std'] = None
            self.parameters[f'tau_xi_{k}_std_inv'] = None
        self.parameters["pi"] = None

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension,
                                                       self.source_dimension, self.ordinal_infos)
        self.attributes.update(['all'], self.parameters)

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'logistic_mixture':
            return self.compute_individual_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name == 'linear':
            return self.compute_individual_tensorized_linear(timepoints, ind_parameters, attribute_type)
        elif self.name == 'mixed_linear-logistic':
            return self.compute_individual_tensorized_mixed(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError("Mutivariate model > Compute individual tensorized")

    def compute_individual_tensorized_linear(self, timepoints, ind_parameters, attribute_type=None):
        positions, velocities, mixing_matrix = self._get_attributes(attribute_type)
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Reshaping
        a = tuple([1] * reparametrized_time.ndimension())
        velocities = velocities.unsqueeze(0).repeat(*tuple(reparametrized_time.shape), 1)
        positions = positions.unsqueeze(0).repeat(*tuple(reparametrized_time.shape), 1)
        reparametrized_time = reparametrized_time.unsqueeze(-1).repeat(*a, velocities.shape[-1])

        # Computation
        LL = velocities * reparametrized_time + positions

        if self.source_dimension != 0:
            sources = ind_parameters['sources']
            wi = torch.nn.functional.linear(sources, mixing_matrix, bias=None)
            LL += wi.unsqueeze(-2)
        return LL

    @staticmethod
    def time_reparametrization_ordinal(timepoints, xi, tau, deltas):
        times = timepoints - tau
        times = times.unsqueeze(-1).unsqueeze(-1) # (ind, timepoints, features, max_level)
        deltas_ = torch.cat([torch.zeros((deltas.shape[0], 1)), deltas], dim=1) # (features, max_level)
        deltas_ = deltas_.unsqueeze(0).unsqueeze(0)
        times = times - deltas_.cumsum(dim=-1)
        return torch.exp(xi).unsqueeze(-1).unsqueeze(-1) * times

    def _get_deltas(self, attribute_type):
        if attribute_type is None:
            return self.attributes.get_deltas()
        elif attribute_type == 'MCMC':
            return self.MCMC_toolbox['attributes'].get_deltas()
        else:
            raise ValueError("The specified attribute type does not exist : {}".format(attribute_type))

    def compute_individual_tensorized_logistic(self, timepoints, individual_parameters, attribute_type=None):
        return self.compute_moments_model(timepoints, individual_parameters, order=0, selected_variables = "all", attribute_type=attribute_type)

    def compute_individual_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):
        raise NotImplementedError()

    def compute_jacobian_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'logistic':
            return self.compute_jacobian_tensorized_logistic(timepoints, ind_parameters, attribute_type)
        elif self.name == 'linear':
            return self.compute_jacobian_tensorized_linear(timepoints, ind_parameters, attribute_type)
        elif self.name == 'mixed_linear-logistic':
            return self.compute_jacobian_tensorized_mixed(timepoints, ind_parameters, attribute_type)
        else:
            raise ValueError("Mutivariate model > Compute jacobian tensorized")

    def compute_jacobian_tensorized_linear(self, timepoints, ind_parameters, attribute_type=None):
        return NotImplementedError()

    def compute_jacobian_tensorized_logistic(self, timepoints, individual_parameters, attribute_type=None):
        '''

        Parameters
        ----------
        timepoints
        ind_parameters
        attribute_type

        Returns
        -------
        The Jacobian of the model with parameters order : [xi, tau, sources].
        This function aims to be used in scipy_minimize.

        '''
        return self.compute_moments_model(timepoints, individual_parameters, order=1, selected_variables = "ind", attribute_type=attribute_type)[1]

    def compute_jacobian_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):
        raise NotImplementedError()

    def compute_moments_model(self, timepoints, individual_parameters, order=0, selected_variables = "all", attribute_type=None):
        """
        Only for logistic model as of now
        Parameters
        ----------
        timepoints
        individual_parameters
        order : order of derivatives to compute.
        attribute_type

        Returns
        If order == 0 : (values,) Model values at the provided timepoints with given individual parameters
        If order == 1 : (values, derivatives,) Model values and derivatives for the selected_variables
        If order == 2 : (values, derivatives, Fisher, ) Model values and derivatives and Fisher information matrix
                                                        for the selected_variables relative to the loss
        -------

        """
        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = individual_parameters['tau_xi'][...,1:], individual_parameters['tau_xi'][...,:1]

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau).unsqueeze(-1)

        denom = v0 * reparametrized_time
        if self.source_dimension != 0:
            sources = individual_parameters['sources']
            wi = sources.matmul(a_matrix.t()).unsqueeze(-2)
            if self.loss == 'ordinal':
                # add an extra dimension for the levels of the ordinal item
                denom = denom.unsqueeze(-1)
                wi = wi.unsqueeze(-1)
                g = g.unsqueeze(-1)
                b = b.unsqueeze(-1)

                deltas = self._get_deltas(attribute_type)
                ordinal_scale = deltas.shape[-1] + 1
                deltas_ = torch.cat([torch.zeros((deltas.shape[0], 1)), deltas], dim=1)  # (features, max_level)
                deltas_ = deltas_.unsqueeze(0).unsqueeze(0)  # add (ind, timepoints) dimensions
                denom = denom - deltas_.cumsum(dim=-1)
            denom += wi
        model_values = 1. / (1. + g * torch.exp(-denom * b))

        if order == 0:
            return model_values

        derivatives = {}
        d = (model_values * (1. - model_values)).unsqueeze(-1)
        c = d * b.unsqueeze(-1)

        if selected_variables in ["all", "ind"] or "xi" in selected_variables:
            xi_derivative = (v0 * reparametrized_time).unsqueeze(-1)
            if self.loss == 'ordinal':
                xi_derivative = xi_derivative.unsqueeze(-2).repeat(1, 1, 1, ordinal_scale, 1)
            derivatives["xi"] = c * xi_derivative
        if selected_variables in ["all", "ind"] or "tau" in selected_variables:
            tau_derivative = (-v0 * torch.exp(xi).reshape(-1, 1, 1)).unsqueeze(-1)
            if self.loss == 'ordinal':
                tau_derivative = tau_derivative.unsqueeze(-2).repeat(1, 1, 1, ordinal_scale, 1)
            derivatives["tau"] = c * tau_derivative
        if self.source_dimension != 0 and (selected_variables in ["all", "ind"] or "sources" in selected_variables):
            sources_derivative = c * a_matrix.expand((1, 1, -1, -1))
            if self.loss == 'ordinal':
                sources_derivative = sources_derivative.unsqueeze(-2).repeat(1, 1, 1, ordinal_scale, 1)
            derivatives["sources"] = sources_derivative
        if selected_variables in ["all", "pop"] or "g" in selected_variables:
            g_derivative = -(1 - denom * (g - 1. / g)).unsqueeze(-1)
            derivatives["g"] = d * g_derivative.repeat(1, 1, 1, self.dimension) * torch.eye(self.dimension).expand(1, 1, -1, -1)
        if selected_variables in ["all", "pop"] or "v0" in selected_variables:
            if self.source_dimension != 0:
                # computing derivatives of Householder method
                e1 = torch.zeros(self.dimension)
                e1[0] = 1
                sign = torch.sign(v0[0])
                norm_v0 = torch.norm(v0)
                alpha = sign * norm_v0
                u_vector = v0 - alpha * e1
                norm_u = torch.norm(u_vector)
                v_vector = u_vector / norm_u
                v_vector = v_vector.reshape(1, -1)
                # q_matrix = torch.eye(self.dimension) - 2 * v_vector.permute(1, 0) * v_vector
                first_comp = torch.zeros((self.dimension, self.dimension))
                first_comp[0, :] = sign * v0 / norm_v0
                first_term = (torch.eye(self.dimension) - first_comp) * norm_u
                second_term = u_vector + (1 - sign * v0[0] / norm_v0) * v0
                second_term = (u_vector.unsqueeze(-1) / norm_u) * second_term.unsqueeze(0)
                v_derivative = (first_term - second_term) / (norm_u * norm_u)
                v_derivative = v_derivative.unsqueeze(0)
                v_dv = v_vector.unsqueeze(-1) * v_derivative
                q_derivative = v_dv + v_dv.permute(1, 0, 2)
                betas = self.MCMC_toolbox['attributes'].betas
                householder_term = sources.matmul(betas.T).matmul(q_derivative[:, 1:]).permute(1, 0, 2).unsqueeze(1)
                time_term = reparametrized_time.unsqueeze(-1) * torch.eye(self.dimension).expand(1, 1, -1, -1)
                v0_derivative = v0 * (time_term + householder_term)
            else:
                v0_derivative = (v0 * reparametrized_time).unsqueeze(-1)
            if self.loss == 'ordinal':
                v0_derivative = v0_derivative.unsqueeze(-2).repeat(1, 1, 1, ordinal_scale, 1)
            derivatives["v0"] = c * v0_derivative
        if self.source_dimension != 0 and (selected_variables in ["all", "pop"] or "betas" in selected_variables):
            betas_derivative = sources.reshape(-1, self.source_dimension, 1, 1) * self.MCMC_toolbox[
                'attributes'].orthonormal_basis.expand(1, 1, -1, -1)
            betas_derivative = betas_derivative.permute(0, 2, 3, 1)
            s = model_values.shape
            betas_derivative = betas_derivative.reshape(s[0], 1, s[2], -1)
            if self.loss == 'ordinal':
                betas_derivative = betas_derivative.unsqueeze(-2).repeat(1, 1, 1, ordinal_scale, 1)
            derivatives["betas"] = c * betas_derivative

        if self.loss == 'ordinal' and (selected_variables in ["all", "pop"] or "deltas" in selected_variables):
            deltas_derivative = torch.tril(deltas.expand(-1, -1, ordinal_scale))
            derivatives["deltas"] = c.unsqueeze(-1) * deltas_derivative.reshape(1, 1, -1, -1, -1)

        if order == 1:
            return model_values, derivatives

        gradient = torch.cat([v for k, v in derivatives.items()], dim=-1).unsqueeze(-1)
        fisher = gradient * gradient.transpose(-1, -2)

        if self.loss == "MSE":
            fisher = fisher / (self.parameters['noise_std'].reshape(1, 1, -1, 1, 1))**2

        elif self.loss == "crossentropy":
            fisher = fisher / d.unsqueeze(-1)

        elif self.loss == "ordinal":
            s = list(model_values.shape)
            s[-1] = 1
            eta_i = torch.cat([torch.ones(s), model_values], -1)
            eta_i_plus_1 = torch.cat([model_values, torch.zeros(s)], -1)
            fisher_loss = eta_i * eta_i_plus_1 * (eta_i - eta_i_plus_1)
            fisher_loss = fisher_loss.sum(dim=-1) / (d ** 2)
            fisher = fisher * fisher_loss.unsqueeze(-1)

        if order == 2:
            return model_values, derivatives, fisher

    def compute_regularity_realization(self, realization, cluster=0, proba_clusters=None):
        # Instantiate torch distribution
        std_inv = None
        if realization.variable_type == 'population':
            mean = self.parameters[realization.name]
            # TODO : Sure it is only MCMC_toolbox?
            std = self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)]
        elif realization.variable_type == 'individual':
            name = realization.name
            if name == 'tau_xi':
                if proba_clusters is not None:
                    regs = torch.stack([self.compute_regularity_realization(realization, cluster=k) for k in
                                       range(self.nb_clusters)])
                    reg = (regs * proba_clusters.unsqueeze(-1)).sum(dim=0)
                    return reg
                name = name + f'_{cluster}'
                std_inv = self.parameters["{0}_std_inv".format(name)]
            mean = self.parameters["{0}_mean".format(name)]
            std = self.parameters["{0}_std".format(name)]
        else:
            raise ValueError("Variable type not known")

        return self.compute_regularity_variable(realization.tensor_realizations, mean, std, std_inv=std_inv)

    def compute_moments_regularization(self, realizations, order=0, selected_variables="all", attribute_type=None, cluster=0, proba_clusters=None):
        '''

        Parameters
        ----------
        realizations
        order
        selected_variables
        attribute_type

        Returns
        -------
        Moments of regularization in negative LL, works for population only during MCMC fit

        '''
        if selected_variables == "all":
            selected_variables = ["xi", "tau", "g", "v0"]
            if self.source_dimension != 0:
                selected_variables.insert(2, "sources")
                selected_variables.append("betas")
            if self.loss == 'ordinal':
                selected_variables.append("deltas")
        elif selected_variables == "pop":
            selected_variables = ["g", "v0"]
            if self.source_dimension != 0:
                selected_variables.append("betas")
            if self.loss == 'ordinal':
                selected_variables.append("deltas")
        elif selected_variables == "ind":
            selected_variables = ["xi", "tau"]
            if self.source_dimension != 0:
                selected_variables.append("sources")
        regularization = {}
        derivatives = {}
        for key in selected_variables:
            realization = realizations[key]
            values = realization.tensor_realizations
            if realization.variable_type == 'population':
                mean = self.parameters[realization.name]
                # TODO : Sure it is only MCMC_toolbox?
                std = self.MCMC_toolbox['priors']['{0}_std'.format(realization.name)]
                regularization[key] = (
                            (values - mean) ** 2 / (2 * std * std) + 0.5 * math.log(2 * math.pi * std * std)).reshape(
                    -1)
                derivatives[key] = ((values - mean) / (std * std)).reshape(-1)
            elif realization.variable_type == 'individual':
                name = realization.name
                if name in ['xi', 'tau']:
                    name = name+f'_{cluster}'
                mean = self.parameters["{0}_mean".format(name)]
                std = self.parameters["{0}_std".format(name)]
                regularization[key] = (
                            (values - mean) ** 2 / (2 * std * std) + 0.5 * math.log(2 * math.pi * std * std)).reshape(values.shape[0],
                    -1)
                derivatives[key] = ((values - mean) / (std * std)).reshape(values.shape[0], -1)
            else:
                raise ValueError("Variable type not known")

        if order == 0:
            return regularization
        if order == 1:
            return regularization, derivatives
        if order == 2:
            fish = [v * v for k, v in derivatives.items()]
            fisher = torch.cat(fish, dim=-1)
            fisher_matrix = fisher.unsqueeze(-1) * fisher.unsqueeze(-2)
            return regularization, derivatives, fisher_matrix

    def compute_loss_derivative(self, data, parameter_values, model_values, model_derivatives,
                                regularization_derivatives, sum_individuals=True):
        '''

        Parameters
        ----------
        data
        parameter_values
        model_values
        model_derivative
        regularization_derivative

        Returns
        -------

        '''
        values = data.values.unsqueeze(-1)
        attachment = model_values.unsqueeze(-1)
        derivatives = {}

        if self.loss == 'MSE':
            diff = attachment - values
            for key in model_derivatives:
                jacobian = model_derivatives[key]
                attachment = diff * jacobian
                mask = (attachment != attachment)
                attachment[mask] = 0.
                # Set nan to zero, not to count in the sum
                attachment = torch.sum(attachment, dim=(1, 2)) / (self.parameters['noise_std'] ** 2)
                if sum_individuals:
                    attachment = attachment.sum(dim=0)
                derivatives[key] = attachment
        elif self.loss == 'crossentropy':
            diff = attachment - values
            attachment = torch.clamp(attachment, 1e-38, 1. - 1e-7)  # safety before dividing
            neg_crossentropy = diff / (attachment * (1. - attachment))
            for key in model_derivatives:
                jacobian = model_derivatives[key]
                neg_crossentropy = neg_crossentropy * jacobian
                mask = (neg_crossentropy != neg_crossentropy)
                neg_crossentropy[mask] = 0.  # Set nan to zero, not to count in the sum
                attachment = torch.sum(neg_crossentropy, dim=(1, 2))
                if sum_individuals:
                    attachment = attachment.sum(dim=0)
                derivatives[key] = attachment
        elif self.loss == 'ordinal':
            max_level = max([feat["nb_levels"] for feat in self.ordinal_infos])

            s = list(attachment.shape)
            s[-1] = 1
            ones = torch.ones(size=tuple(s)).float()

            pred = torch.cat([ones, attachment], dim=-1)
            vals = values.long().clamp(0, max_level)
            t = torch.eye(max_level + 1)
            for k in range(max_level):
                t[k, k + 1] = -1.
            vals[vals != vals] = 0
            vals = t[vals]

            mask = (values == values)

            for key in model_derivatives:
                jacobian = model_derivatives[key]
                s = list(jacobian.shape)
                s[-1] = 1
                zeros = torch.zeros(size=tuple(s)).float()
                jac = torch.cat([zeros, jacobian], dim=-1)

                LL = - (jac * vals).sum(dim=-1) / (pred * vals).sum(dim=-1)
                attachment = torch.sum(mask.float() * LL, dim=(1, 2))
                if sum_individuals:
                    attachment = attachment.sum(dim=0)
                derivatives[key] = attachment

        # Regularity
        regularities = []
        for key in regularization_derivatives:
            derivatives[key] = derivatives[key] + regularization_derivatives[key]

        return derivatives



    """
    def compute_individual_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):


        raise ValueError("Do not use !!!")

        # Hyperparameters : split # TODO
        split = 1
        idx_linear = list(range(split))
        idx_logistic = list(range(split, self.dimension))

        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau, sources = ind_parameters['xi'], ind_parameters['tau'], ind_parameters['sources']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.reshape(*timepoints.shape, 1)
        v0 = v0.reshape(1, 1, -1)

        LL = v0 * reparametrized_time
        if self.source_dimension != 0:
            wi = sources.matmul(a_matrix.t())
            LL += wi.unsqueeze(-2)

        # Logistic Part
        LL_log = 1. + g * torch.exp(-LL * b)
        model_logistic = (1. / LL_log)[:,:,idx_logistic]

        # Linear Part
        model_linear = (LL + torch.log(g))[:,:,idx_linear]

        # Concat
        model = torch.cat([model_linear, model_logistic], dim=2)

        return model"""
    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 0.01, 'v0_std': 0.01, 'betas_std': 0.01},
            'attributes': AttributesFactory.attributes(self.name, self.dimension,
                                                       self.source_dimension, self.ordinal_infos)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # TODO maybe not here
        # Initialize priors
        self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone()
        self.MCMC_toolbox['priors']['s_v0'] = 0.1
        if self.loss == 'ordinal':
            self.MCMC_toolbox['priors']['deltas_std'] = 0.01

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('v0', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations
        if any(c in L for c in ('deltas', 'all')) and self.loss=='ordinal':
            values['deltas'] = realizations['deltas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['tau_xi'].tensor_realizations[..., 1])
        realizations['tau_xi'].tensor_realizations[..., 1] = realizations['tau_xi'].tensor_realizations[..., 1] - mean_xi
        realizations['v0'].tensor_realizations = realizations['v0'].tensor_realizations + mean_xi

        self.update_MCMC_toolbox(['all'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations, clusters):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'v0': realizations['v0'].tensor_realizations,
            'tau_xi': realizations['tau_xi'].tensor_realizations,
        }

        sufficient_statistics["proba_clusters"] = clusters

        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations

        if self.loss == 'ordinal':
            sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations
        ind_parameters = self.get_param_from_real(realizations)


        if self.loss == 'MSE':
            data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                     ind_parameters,
                                                                     attribute_type='MCMC')
            norm_0 = data.values * data.values * data.mask.float()
            norm_1 = data_reconstruction * (data.values * data.mask.float()).unsqueeze(0)
            norm_2 = data_reconstruction * data_reconstruction * data.mask.float().unsqueeze(0)
            sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=-1)
            sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=-1)
            sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=-1)

        elif self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")
        elif self.loss == 'ordinal':
            sufficient_statistics['log-likelihood'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations, clusters=None):
        # if self.name == 'logistic':

        # try without centering during burn-in
#        realizations = self._center_xi_realizations(realizations)

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations

        if self.MCMC_toolbox['priors']['v0_mean'] is not None:
            v0_mean = self.MCMC_toolbox['priors']['v0_mean']
            v0_emp = realizations['v0'].tensor_realizations
            s_v0 = self.MCMC_toolbox['priors']['s_v0']
            sigma_v0 = self.MCMC_toolbox['priors']['v0_std']
            self.parameters['v0'] = (1 / (1 / (s_v0 ** 2) + 1 / (sigma_v0 ** 2))) * (
                        v0_emp / (sigma_v0 ** 2) + v0_mean / (s_v0 ** 2))
        else:
            self.parameters['v0'] = realizations['v0'].tensor_realizations

        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations

        if self.loss == 'ordinal':
            self.parameters['deltas'] = realizations['deltas'].tensor_realizations
            # Stochastic sufficient statistics used to update the parameters of the model
        tau_xi = realizations['tau_xi'].tensor_realizations
        for k in range(self.nb_clusters):
            cluster = clusters[k]
            if cluster.sum() != 0.:
                S_inv = 1./cluster.sum()
                cluster = cluster.unsqueeze(-1)
                self.parameters[f'tau_xi_{k}_mean'] = S_inv * (cluster * tau_xi).sum(dim=0)
                err = tau_xi - self.parameters[f'tau_xi_{k}_mean'].unsqueeze(0)
                err2 = (cluster * err).T @ err
                tau_xi_std = S_inv * err2
                self.parameters[f'tau_xi_{k}_std'] = tau_xi_std + 1e-8 * torch.eye(2)
                self.parameters[f'tau_xi_{k}_std_inv'] = torch.linalg.inv(self.parameters[f'tau_xi_{k}_std'])

        self.parameters['pi'] = clusters.sum(dim=1)/clusters.sum()

        param_ind = self.get_param_from_real(realizations)

        if self.loss == 'MSE':
            squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type='MCMC').sum()
            self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

        elif self.loss == 'crossentropy':
            crossentropy = (clusters.squeeze(-1)*self.compute_individual_attachment_tensorized(data, param_ind,
                                                                         attribute_type='MCMC')).sum()
            self.parameters['crossentropy'] = crossentropy

        elif self.loss == 'ordinal':
            crossentropy = (clusters.squeeze(-1)*self.compute_individual_attachment_tensorized(data, param_ind,
                                                                         attribute_type='MCMC')).sum()
            self.parameters['log-likelihood'] = crossentropy

        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats):
        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        self.parameters['g'] = suff_stats['g']
        self.parameters['v0'] = suff_stats['v0']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']
        if self.loss == 'ordinal':
            self.parameters['deltas'] = suff_stats['deltas']

        clusters = suff_stats["proba_clusters"]

        for k in range(self.nb_clusters):
            cluster = clusters[k]
            if cluster.sum() != 0.:
                S_inv = 1./cluster.sum()
                cluster = cluster.unsqueeze(-1)

                tau_xi = suff_stats["tau_xi"]
                self.parameters[f'tau_xi_{k}_mean'] = S_inv * (cluster * tau_xi).sum(dim=0)
                err = tau_xi - self.parameters[f'tau_xi_{k}_mean'].unsqueeze(0)
                err2 = (cluster * err).T @ err
                tau_xi_std = S_inv * err2
                self.parameters[f'tau_xi_{k}_std'] = tau_xi_std + 1e-8 * torch.eye(2)
                self.parameters[f'tau_xi_{k}_std_inv'] = torch.linalg.inv(self.parameters[f'tau_xi_{k}_std'])

        self.parameters["pi"] = clusters.sum(dim=1)/clusters.sum()
        if self.loss == 'MSE':
            S1 = torch.sum(suff_stats['obs_x_obs'])
            S2 = torch.sum(suff_stats['obs_x_reconstruction'])
            S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / (data.mask.float()).sum())

        elif self.loss == 'crossentropy':
            self.parameters['crossentropy'] = (clusters.squeeze(-1)*suff_stats['crossentropy']).sum()

        elif self.loss == 'ordinal':
            self.parameters['log-likelihood'] = (clusters.squeeze(-1)*suff_stats['log-likelihood']).sum()

    ###################################
    ### Random Variable Information ###
    ###################################

    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        v0_infos = {
            "name": "v0",
            "shape": torch.Size([self.dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }

        betas_infos = {
            "name": "betas",
            "shape": torch.Size([self.dimension - 1, self.source_dimension]),
            "type": "population",
            "rv_type": "multigaussian"
        }


        ## Individual variables
        tau_xi_infos = {
            "name": "tau_xi",
            "shape": torch.Size([2]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        sources_infos = {
            "name": "sources",
            "shape": torch.Size([self.source_dimension]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        variables_infos = {
            "g": g_infos,
            "v0": v0_infos,
            "tau_xi": tau_xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        if self.loss == 'ordinal':
            max_level = max([feat["nb_levels"] for feat in self.ordinal_infos])
            deltas_infos = {
                "name": "deltas",
                "shape": torch.Size([self.dimension, max_level - 1]),
                "type": "population",
                "rv_type": "multigaussian"
            }
            variables_infos['deltas'] = deltas_infos

        return variables_infos

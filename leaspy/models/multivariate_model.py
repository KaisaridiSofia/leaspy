import torch

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_factory import AttributesFactory


class MultivariateModel(AbstractMultivariateModel):
    def __init__(self, name):
        super(MultivariateModel, self).__init__(name)
        self.parameters["v0"] = None
        self.MCMC_toolbox['priors']['v0_std'] = None  # Value, Coef

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        if self.name == 'logistic':
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

    def compute_individual_tensorized_logistic(self, timepoints, ind_parameters, attribute_type=None):
        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.reshape(*timepoints.shape, 1)
        v0 = v0.reshape(1, 1, -1)

        LL = v0 * reparametrized_time
        if self.source_dimension != 0:
            sources = ind_parameters['sources']
            wi = sources.matmul(a_matrix.t())
            LL += wi.unsqueeze(-2)
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL
        return model

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

    def compute_jacobian_tensorized_logistic(self, timepoints, ind_parameters, attribute_type=None):
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
        # Population parameters
        g, v0, a_matrix = self._get_attributes(attribute_type)
        g_plus_1 = 1. + g
        b = g_plus_1 * g_plus_1 / g

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        reparametrized_time = reparametrized_time.reshape(*timepoints.shape, 1)
        v0 = v0.reshape(1, 1, -1)

        LL = v0 * reparametrized_time
        if self.source_dimension != 0:
            sources = ind_parameters['sources']
            wi = sources.matmul(a_matrix.t())
            LL += wi.unsqueeze(-2)
        LL = 1. + g * torch.exp(-LL * b)
        model = 1. / LL

        c = model * (1. - model) * b

        xi_derivative = (v0 * reparametrized_time).unsqueeze(1)
        tau_derivative = (-v0 * torch.exp(xi) * torch.ones_like(reparametrized_time)).unsqueeze(1)
        sources_derivative = a_matrix.t().unsqueeze(1).unsqueeze(0).repeat(xi_derivative.shape[0], 1, xi_derivative.shape[2],1)
        jacob = c * torch.cat([xi_derivative, tau_derivative, sources_derivative], 1)

        return jacob

    def compute_jacobian_tensorized_mixed(self, timepoints, ind_parameters, attribute_type=None):
        raise NotImplementedError()

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
            'attributes': AttributesFactory.attributes(self.name, self.dimension, self.source_dimension)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

        # TODO maybe not here
        # Initialize priors
        self.MCMC_toolbox['priors']['v0_mean'] = self.parameters['v0'].clone()
        self.MCMC_toolbox['priors']['s_v0'] = 0.1

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('v0', 'all')):
            values['v0'] = realizations['v0'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations

        self.MCMC_toolbox['attributes'].update(name_of_the_variables_that_have_been_changed, values)

    def _center_xi_realizations(self, realizations):
        mean_xi = torch.mean(realizations['xi'].tensor_realizations)
        realizations['xi'].tensor_realizations = realizations['xi'].tensor_realizations - mean_xi
        realizations['v0'].tensor_realizations = realizations['v0'].tensor_realizations + mean_xi

        self.update_MCMC_toolbox(['all'], realizations)
        return realizations

    def compute_sufficient_statistics(self, data, realizations):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

        sufficient_statistics = {
            'g': realizations['g'].tensor_realizations,
            'v0': realizations['v0'].tensor_realizations,
            'tau': realizations['tau'].tensor_realizations,
            'tau_sqrd': torch.pow(realizations['tau'].tensor_realizations, 2),
            'xi': realizations['xi'].tensor_realizations,
            'xi_sqrd': torch.pow(realizations['xi'].tensor_realizations, 2)
        }
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations

        ind_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 ind_parameters,
                                                                 attribute_type='MCMC')
        norm_0 = data.values * data.values * data.mask.float()
        norm_1 = data.values * data_reconstruction * data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction * data.mask.float()
        sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=2)

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations, clusters=None):
        # if self.name == 'logistic':
        realizations = self._center_xi_realizations(realizations)

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

        xi = realizations['xi'].tensor_realizations
        tau = realizations['tau'].tensor_realizations
        if clusters is None:
            self.parameters['xi_std'] = torch.std(xi)
            self.parameters['tau_mean'] = torch.mean(tau)
            self.parameters['tau_std'] = torch.std(tau)

            param_ind = self.get_param_from_real(realizations)
            squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type='MCMC').sum()
            self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

            if self.loss == 'crossentropy':
                crossentropy = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type='MCMC').sum()
                self.parameters['crossentropy'] = crossentropy
            # Stochastic sufficient statistics used to update the parameters of the model
        else:
            if clusters.sum() != 0.:
                S_inv = 1./clusters.sum()
                clusters = clusters.unsqueeze(-1)
                xi_err = xi
                xi_err2 = xi_err * xi_err
                self.parameters['xi_std'] = torch.sqrt((S_inv * (clusters * xi_err2)).sum())
                self.parameters['tau_mean'] = S_inv * (clusters * tau).sum()
                tau_err = tau - self.parameters['tau_mean']
                tau_err2 = tau_err * tau_err
                self.parameters['tau_std'] = torch.sqrt((S_inv * (clusters * tau_err2)).sum())

                param_ind = self.get_param_from_real(realizations)
                squared_diff = (clusters.squeeze(-1)*self.compute_sum_squared_tensorized(data, param_ind, attribute_type='MCMC')).sum()
                self.parameters['noise_std'] = torch.sqrt(squared_diff / ((clusters.unsqueeze(-1) * data.mask.float())).sum())

                if self.loss == 'crossentropy':
                    crossentropy = (clusters.squeeze(-1)*self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                 attribute_type='MCMC')).sum()
                    self.parameters['crossentropy'] = crossentropy

        # Stochastic sufficient statistics used to update the parameters of the model

    def update_model_parameters_normal(self, data, suff_stats, clusters=None):
        # TODO with Raphael : check the SS, especially the issue with mean(xi) and v_k
        # TODO : 1. Learn the mean of xi and v_k
        # TODO : 2. Set the mean of xi to 0 and add it to the mean of V_k
        self.parameters['g'] = suff_stats['g']
        self.parameters['v0'] = suff_stats['v0']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        if clusters is None:

            tau_mean = self.parameters['tau_mean']
            tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
            self.parameters['tau_std'] = torch.sqrt(torch.clamp(tau_std_updt + tau_mean ** 2, 1e-32))
            self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

            xi_std_updt = torch.mean(suff_stats['xi_sqrd'])
            self.parameters['xi_std'] = torch.sqrt(torch.clamp(xi_std_updt, 1e-32))

            S1 = data.L2_norm
            S2 = torch.sum(suff_stats['obs_x_reconstruction'])
            S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

            if self.loss == 'crossentropy':
                self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

        else:
            if clusters.sum() != 0.:
                S_inv = 1./clusters.sum()
                clusters = clusters.unsqueeze(-1)

                tau_mean = self.parameters['tau_mean']
                tau_std_updt = S_inv*(clusters*(
                        suff_stats['tau_sqrd'] - 2 * tau_mean * suff_stats['tau'] + tau_mean ** 2)).sum()
                self.parameters['tau_std'] = torch.sqrt(torch.clamp(tau_std_updt, 1e-32))
                self.parameters['tau_mean'] = S_inv*(clusters*suff_stats['tau']).sum()

                xi_std_updt = S_inv * (clusters * suff_stats['xi_sqrd']).sum()
                self.parameters['xi_std'] = torch.sqrt(torch.clamp(xi_std_updt, 1e-32))

                S1 = torch.sum(clusters*suff_stats['obs_x_obs'])
                S2 = torch.sum(clusters*suff_stats['obs_x_reconstruction'])
                S3 = torch.sum(clusters*suff_stats['reconstruction_x_reconstruction'])

                self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / ((clusters.unsqueeze(-1) * data.mask.float()).sum()))

                if self.loss == 'crossentropy':
                    self.parameters['crossentropy'] = (clusters.squeeze(-1)*suff_stats['crossentropy']).sum()

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
        tau_infos = {
            "name": "tau",
            "shape": torch.Size([1]),
            "type": "individual",
            "rv_type": "gaussian"
        }

        xi_infos = {
            "name": "xi",
            "shape": torch.Size([1]),
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
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos
        return variables_infos

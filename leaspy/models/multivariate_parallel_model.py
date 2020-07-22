import torch

from .abstract_multivariate_model import AbstractMultivariateModel
from .utils.attributes.attributes_logistic_parallel import AttributesLogisticParallel
#from .utils.initialization.model_initialization import initialize_logistic_parallel # not used


class MultivariateParallelModel(AbstractMultivariateModel):
    def __init__(self, name):
        super(MultivariateParallelModel, self).__init__(name)
        self.parameters["deltas"] = None
        self.MCMC_toolbox['priors']['deltas_std'] = None

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            if k in ['mixing_matrix']:
                continue
            self.parameters[k] = torch.tensor(parameters[k], dtype=torch.float32)
        self.attributes = AttributesLogisticParallel(self.dimension, self.source_dimension)
        self.attributes.update(['all'], self.parameters)

    def compute_individual_tensorized(self, timepoints, ind_parameters, attribute_type=None):
        # Population parameters
        g, deltas, a_matrix = self._get_attributes(attribute_type)
        deltas_exp = torch.exp(-deltas)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        LL = deltas.unsqueeze(0).repeat(timepoints.shape[0], 1)
        if self.source_dimension != 0:
            sources = ind_parameters['sources']
            wi = torch.nn.functional.linear(sources, a_matrix, bias=None)
            LL += wi * (g * deltas_exp + 1) ** 2 / (g * deltas_exp)
        LL = -reparametrized_time.unsqueeze(-1) - LL.unsqueeze(-2)
        model = 1. / (1. + g * torch.exp(LL))

        return model

    ##############################
    ### MCMC-related functions ###
    ##############################

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 1., 'deltas_std': 0.1, 'betas_std': 0.1},
            'attributes': AttributesLogisticParallel(self.dimension, self.source_dimension)
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('deltas', 'all')):
            values['deltas'] = realizations['deltas'].tensor_realizations
        if any(c in L for c in ('betas', 'all')) and self.source_dimension != 0:
            values['betas'] = realizations['betas'].tensor_realizations
        if any(c in L for c in ('xi_mean', 'all')):
            values['xi_mean'] = self.parameters['xi_mean']

        self.MCMC_toolbox['attributes'].update(L, values)

    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach()
        sufficient_statistics['deltas'] = realizations['deltas'].tensor_realizations.detach()
        if self.source_dimension != 0:
            sufficient_statistics['betas'] = realizations['betas'].tensor_realizations.detach()
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        ind_parameters = self.get_param_from_real(realizations)

        data_reconstruction = self.compute_individual_tensorized(data.timepoints,
                                                                 ind_parameters,
                                                                 attribute_type='MCMC')
        data_reconstruction *= data.mask.float()
        norm_1 = data.values * data_reconstruction * data.mask.float()
        norm_2 = data_reconstruction * data_reconstruction * data.mask.float()
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=2)

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data, ind_parameters,
                                                                                                  attribute_type="MCMC")

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations):

        self.parameters['g'] = realizations['g'].tensor_realizations.detach()
        self.parameters['deltas'] = realizations['deltas'].tensor_realizations.detach()
        if self.source_dimension != 0:
            self.parameters['betas'] = realizations['betas'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        self.parameters['xi_mean'] = torch.mean(xi)
        self.parameters['xi_std'] = torch.std(xi)
        tau = realizations['tau'].tensor_realizations.detach()
        self.parameters['tau_mean'] = torch.mean(tau)
        self.parameters['tau_std'] = torch.std(tau)

        param_ind = self.get_param_from_real(realizations)
        data_fit = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type='MCMC')
        data_fit *= data.mask.float()
        squared_diff = ((data_fit - data.values) ** 2).sum()
        squared_diff = squared_diff.detach()  # Remove the gradients
        self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)
        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                            attribute_type="MCMC").sum()
    def update_model_parameters_normal(self, data, suff_stats):

        self.parameters['g'] = suff_stats['g']
        self.parameters['deltas'] = suff_stats['deltas']
        if self.source_dimension != 0:
            self.parameters['betas'] = suff_stats['betas']

        tau_mean = self.parameters['tau_mean']
        tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
        self.parameters['tau_std'] = torch.sqrt(torch.clamp(tau_std_updt + self.parameters['tau_mean'] ** 2, 0.))
        self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])

        xi_mean = self.parameters['xi_mean']
        xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
        self.parameters['xi_std'] = torch.sqrt(torch.clamp(xi_std_updt + self.parameters['xi_mean'] ** 2, 0.))
        self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])

        S1 = data.L2_norm
        S2 = torch.sum(suff_stats['obs_x_reconstruction'])
        S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

        self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

        if self.loss == 'crossentropy':
            self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

    ###################################
    ### Random Variable Information ###
    ###################################

    def random_variable_informations(self):
        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([1]),
            "type": "population",
            "rv_type": "multigaussian"
        }
        deltas_infos = {
            "name": "deltas",
            "shape": torch.Size([self.dimension - 1]),
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
            "deltas": deltas_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        if self.source_dimension != 0:
            variables_infos['sources'] = sources_infos
            variables_infos['betas'] = betas_infos

        return variables_infos

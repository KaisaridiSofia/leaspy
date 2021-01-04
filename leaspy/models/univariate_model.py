import json

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import torch

from .abstract_model import AbstractModel
from .utils.attributes.attributes_univariate import AttributesUnivariate


class UnivariateModel(AbstractModel):
    ###########################
    ## Initialization
    ###########################
    def __init__(self, name):
        super(UnivariateModel, self).__init__(name)
        self.dimension = 1
        self.source_dimension = 0  # TODO, None ???
        self.parameters = {
            "g": None,
            "tau_mean": None, "tau_std": None,
            "xi_mean": None, "xi_std": None,
            "noise_std": None
        }
        self.bayesian_priors = None
        self.attributes = None

        # MCMC related "parameters"
        self.MCMC_toolbox = {
            'attributes': None,
            'priors': {
                'g_std': None,  # tq p0 = 1 / (1+exp(g)) i.e. g = 1/p0 - 1
            }
        }

    def save(self, path):
        model_parameters_save = self.parameters.copy()
        for key, value in model_parameters_save.items():
            if type(value) in [torch.Tensor]:
                model_parameters_save[key] = value.tolist()
        model_settings = {
            'name': 'univariate',
            'features': self.features,
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp)

    def load_hyperparameters(self, hyperparameters):
        if 'features' in hyperparameters.keys():
            self.features = hyperparameters['features']
        return

    def initialize(self, dataset):

        # "Smart" initialization : may be improved
        # TODO !
        self.features = dataset.headers
        self.parameters = {
            'g': [1.],
            'tau_mean': 70., 'tau_std': 2.,
            'xi_mean': -3., 'xi_std': .1,
            'noise_std': [.1]}
        self.parameters = {key: torch.tensor(val) for key, val in self.parameters.items()}
        self.attributes = AttributesUnivariate()
        self.is_initialized = True

    def load_parameters(self, parameters):
        self.parameters = {}
        for k in parameters.keys():
            self.parameters[k] = torch.tensor(parameters[k])
        self.attributes = AttributesUnivariate()
        self.attributes.update(['all'], self.parameters)

    def initialize_MCMC_toolbox(self):
        self.MCMC_toolbox = {
            'priors': {'g_std': 1.},
            'attributes': AttributesUnivariate()
        }

        population_dictionary = self._create_dictionary_of_population_realizations()
        self.update_MCMC_toolbox(["all"], population_dictionary)

    ##########
    # CORE
    ##########
    def update_MCMC_toolbox(self, name_of_the_variables_that_have_been_changed, realizations):
        L = name_of_the_variables_that_have_been_changed
        values = {}
        if any(c in L for c in ('g', 'all')):
            values['g'] = realizations['g'].tensor_realizations
        if any(c in L for c in ('xi_mean', 'all')):
            values['xi_mean'] = self.parameters['xi_mean']

        self.MCMC_toolbox['attributes'].update(L, values)

    def _get_attributes(self, MCMC):
        if MCMC:
            g = self.MCMC_toolbox['attributes'].positions
        else:
            g = self.attributes.positions
        return g

    # def compute_sum_squared_tensorized(self, data, param_ind, attribute_type):
    #    res = self.compute_individual_tensorized(data.timepoints, param_ind, attribute_type)
    #    res *= data.mask
    #    return torch.sum((res * data.mask - data.values) ** 2, dim=(1, 2))

    # TODO generalize in abstract
    def compute_mean_traj(self, timepoints):
        individual_parameters = {
            'xi': torch.tensor([self.parameters['xi_mean']], dtype=torch.float32),
            'tau': torch.tensor([self.parameters['tau_mean']], dtype=torch.float32),
        }

        return self.compute_individual_tensorized(timepoints, individual_parameters)

    def plot_param_ind(self, path, param_ind):
        pdf = matplotlib.backends.backend_pdf.PdfPages(path)
        fig, ax = plt.subplots(1, 1)
        xi, tau = param_ind
        ax.plot(xi.squeeze(1).detach().tolist(), tau.squeeze(1).detach().tolist(), 'x')
        plt.xlabel('xi')
        plt.ylabel('tau')
        pdf.savefig(fig)
        plt.close()
        pdf.close()

    def compute_individual_tensorized(self, timepoints, ind_parameters, MCMC=False):
        # Population parameters
        g = self._get_attributes(MCMC)
        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']
        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        LL = -reparametrized_time.unsqueeze(-1)
        model = 1. / (1. + g * torch.exp(LL))

        return model

    def compute_jacobian_tensorized(self, timepoints, ind_parameters, MCMC=False):
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
        g = self._get_attributes(MCMC)

        # Individual parameters
        xi, tau = ind_parameters['xi'], ind_parameters['tau']

        reparametrized_time = self.time_reparametrization(timepoints, xi, tau)

        # Log likelihood computation
        LL = reparametrized_time.unsqueeze(-1)

        LL = 1. + g * torch.exp(-LL)
        model = 1. / LL

        c = model * (1. - model)

        xi_derivative = reparametrized_time.unsqueeze(1).unsqueeze(-1)
        tau_derivative = (-torch.exp(xi) * torch.ones_like(reparametrized_time)).unsqueeze(1).unsqueeze(-1)

        jacob = c * torch.cat([xi_derivative, tau_derivative], 1)

        return jacob

    def compute_sufficient_statistics(self, data, realizations):
        sufficient_statistics = {}
        sufficient_statistics['g'] = realizations['g'].tensor_realizations.detach() # avoid 0D / 1D tensors mix
        sufficient_statistics['tau'] = realizations['tau'].tensor_realizations
        sufficient_statistics['tau_sqrd'] = torch.pow(realizations['tau'].tensor_realizations, 2)
        sufficient_statistics['xi'] = realizations['xi'].tensor_realizations
        sufficient_statistics['xi_sqrd'] = torch.pow(realizations['xi'].tensor_realizations, 2)

        # TODO : Optimize to compute the matrix multiplication only once for the reconstruction
        ind_parameters = self.get_param_from_real(realizations)
        data_reconstruction = self.compute_individual_tensorized(data.timepoints, ind_parameters, MCMC=True)
        data_reconstruction *= data.mask
        norm_0 = data.values * data.values * data.mask
        norm_1 = data.values * data_reconstruction * data.mask
        norm_2 = data_reconstruction * data_reconstruction * data.mask
        sufficient_statistics['obs_x_obs'] = torch.sum(norm_0, dim=2)
        sufficient_statistics['obs_x_reconstruction'] = torch.sum(norm_1, dim=2)
        sufficient_statistics['reconstruction_x_reconstruction'] = torch.sum(norm_2, dim=2)

        if self.loss == 'crossentropy':
            sufficient_statistics['crossentropy'] = self.compute_individual_attachment_tensorized(data,
                                                                    ind_parameters, attribute_type=True).unsqueeze(-1)

        return sufficient_statistics

    def update_model_parameters_burn_in(self, data, realizations, clusters=None):

        # Memoryless part of the algorithm
        self.parameters['g'] = realizations['g'].tensor_realizations.detach()
        xi = realizations['xi'].tensor_realizations.detach()
        tau = realizations['tau'].tensor_realizations.detach()
        if clusters is None:
            self.parameters['xi_mean'] = torch.mean(xi)
            self.parameters['xi_std'] = torch.std(xi)
            self.parameters['tau_mean'] = torch.mean(tau)
            self.parameters['tau_std'] = torch.std(tau)

            param_ind = self.get_param_from_real(realizations)
            squared_diff = self.compute_sum_squared_tensorized(data, param_ind, attribute_type=True).sum()
            self.parameters['noise_std'] = torch.sqrt(squared_diff / data.n_observations)

            if self.loss == 'crossentropy':
                crossentropy = self.compute_individual_attachment_tensorized(data, param_ind, attribute_type=True).sum()
                self.parameters['crossentropy'] = crossentropy
            # Stochastic sufficient statistics used to update the parameters of the model
        else:
            if clusters.sum() != 0.:
                S_inv = 1./clusters.sum()
                clusters = clusters.unsqueeze(-1)
                self.parameters['xi_mean'] = S_inv * (clusters * xi).sum()
                xi_err = xi - self.parameters['xi_mean']
                xi_err2 = xi_err * xi_err
                self.parameters['xi_std'] = torch.sqrt((S_inv * (clusters * xi_err2)).sum())
                self.parameters['tau_mean'] = S_inv * (clusters * tau).sum()
                tau_err = tau - self.parameters['tau_mean']
                tau_err2 = tau_err * tau_err
                self.parameters['tau_std'] = torch.sqrt((S_inv * (clusters * tau_err2)).sum())

                param_ind = self.get_param_from_real(realizations)
                squared_diff = (clusters.squeeze(-1)*self.compute_sum_squared_tensorized(data, param_ind, attribute_type=True)).sum()
                self.parameters['noise_std'] = torch.sqrt(squared_diff / ((clusters.unsqueeze(-1) * data.mask.float()).sum()))

                if self.loss == 'crossentropy':
                    crossentropy = (clusters.squeeze(-1)*self.compute_individual_attachment_tensorized(data, param_ind,
                                                                                 attribute_type=True)).sum()
                    self.parameters['crossentropy'] = crossentropy

    def update_model_parameters_normal(self, data, suff_stats, clusters=None):
        self.parameters['g'] = suff_stats['g']

        if clusters is None:

            self.parameters['tau_mean'] = torch.mean(suff_stats['tau'])
            tau_mean = self.parameters['tau_mean']
            tau_std_updt = torch.mean(suff_stats['tau_sqrd']) - 2 * tau_mean * torch.mean(suff_stats['tau'])
            self.parameters['tau_std'] = torch.sqrt(torch.clamp(tau_std_updt + tau_mean ** 2, 1e-32))

            self.parameters['xi_mean'] = torch.mean(suff_stats['xi'])
            xi_mean = self.parameters['xi_mean']
            xi_std_updt = torch.mean(suff_stats['xi_sqrd']) - 2 * xi_mean * torch.mean(suff_stats['xi'])
            self.parameters['xi_std'] = torch.sqrt(torch.clamp(xi_std_updt + xi_mean ** 2, 1e-32))

            S1 = torch.sum(suff_stats['obs_x_obs'])
            S2 = torch.sum(suff_stats['obs_x_reconstruction'])
            S3 = torch.sum(suff_stats['reconstruction_x_reconstruction'])

            self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / data.n_observations)

            if self.loss == 'crossentropy':
                self.parameters['crossentropy'] = suff_stats['crossentropy'].sum()

        else:
            if clusters.sum() != 0.:
                S_inv = 1./clusters.sum()
                clusters = clusters.unsqueeze(-1)

                self.parameters['tau_mean'] = S_inv * (clusters * suff_stats['tau']).sum()
                tau_mean = self.parameters['tau_mean']
                tau_std_updt = S_inv*(clusters*(
                        suff_stats['tau_sqrd'] - 2 * tau_mean * suff_stats['tau'] + tau_mean ** 2)).sum()
                self.parameters['tau_std'] = torch.sqrt(torch.clamp(tau_std_updt, 1e-32))

                self.parameters['xi_mean'] = S_inv * (clusters * suff_stats['xi']).sum()
                xi_mean = self.parameters['xi_mean']
                xi_std_updt = S_inv * (clusters * (
                            suff_stats['xi_sqrd'] - 2 * xi_mean * suff_stats['xi'] + xi_mean ** 2)).sum()
                self.parameters['xi_std'] = torch.sqrt(torch.clamp(xi_std_updt, 1e-32))

                S1 = torch.sum(clusters*suff_stats['obs_x_obs'])
                S2 = torch.sum(clusters*suff_stats['obs_x_reconstruction'])
                S3 = torch.sum(clusters*suff_stats['reconstruction_x_reconstruction'])

                self.parameters['noise_std'] = torch.sqrt((S1 - 2. * S2 + S3) / ((clusters.unsqueeze(-1) * data.mask.float()).sum()))

                if self.loss == 'crossentropy':
                    self.parameters['crossentropy'] = (clusters*suff_stats['crossentropy']).sum()



    # def get_param_from_real(self,realizations):
    #    xi = realizations['xi'].tensor_realizations
    #    tau = realizations['tau'].tensor_realizations
    #    return (xi,tau)

    def param_ind_from_dict(self, individual_parameters):
        xi, tau = [], []
        for key, item in individual_parameters.items():
            xi.append(item['xi'])
            tau.append(item['tau'])
        xi = torch.tensor(xi).unsqueeze(1)
        tau = torch.tensor(tau).unsqueeze(1)
        return (xi, tau)

    def get_xi_tau(self, param_ind):
        xi, tau = param_ind
        return xi, tau

    def random_variable_informations(self):

        ## Population variables
        g_infos = {
            "name": "g",
            "shape": torch.Size([1]),
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

        variables_infos = {
            "g": g_infos,
            "tau": tau_infos,
            "xi": xi_infos,
        }

        return variables_infos

import torch
import numpy as np

from ..abstract_algo import AbstractAlgo
from ..samplers.hmc_sampler import HMCSampler
from ..samplers.gibbs_sampler import GibbsSampler


class MixtureFitMCMC(AbstractAlgo):

    def __init__(self, settings):

        super().__init__()

        # Algorithm parameters
        self.algo_parameters = settings.parameters
        self.seed = settings.seed
        self.name = 'Mixture_MCMC'

        # Realizations and samplers
        self.realizations = []
        self.task = None
        self.samplers = []
        self.sufficient_statistics = []
        self.current_iteration = 0
        self.nb_clusters = self.algo_parameters['nb_clusters']
        self.pi = np.array([1. / self.nb_clusters for k in range(self.nb_clusters)])
        self.loss = settings.loss
        self.stochastic = False
        if 'stochastic' in self.algo_parameters:
            self.stochastic = self.algo_parameters['stochastic']

        # Annealing
        self.temperature_inv = 1.
        self.temperature = 1.

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, models, realizations, initial_clusters):
        """
        Initialize the samplers, annealing, MCMC toolbox and sufficient statistics.
        :param data:
        :param model:
        :param realizations:
        :return: realizations
        """
        # Samplers
        self._initialize_samplers(models, data)
        self.pi = initial_clusters.sum(axis=1) / float(data.n_individuals)
        for i, model in enumerate(models):
            # MCMC toolbox (cache variables for speed-ups + tricks)
            model.initialize_MCMC_toolbox()
            model.loss = self.loss
        self._initialize_sufficient_statistics(data, models, realizations)
        if self.algo_parameters['annealing']['do_annealing']:
            self._initialize_annealing()
        return realizations

    def _initialize_annealing(self):
        """
        Initialize annealing, setting initial temperature and number of iterations.
        :return:
        """

        # Might introduce a more intelligent initialization depending on the initial acceptance rate

        self._update_temperature()

    def _initialize_samplers(self, models, data):
        """
        Instanciate samplers for Gibbs / HMC sampling as a dictionnary samplers {name: sampler}
        :param model:
        :param data:
        :return:
        """
        for i, model in enumerate(models):
            infos_variables = model.random_variable_informations()
            self.samplers.append(dict.fromkeys(infos_variables.keys()))
            for variable, info in infos_variables.items():
                if info["type"] == "individual":
                    if self.algo_parameters['sampler_ind'] == 'Gibbs':
                        self.samplers[i][variable] = GibbsSampler(info, data.n_individuals)
                    else:
                        self.samplers[i][variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
                else:
                    if self.algo_parameters['sampler_pop'] == 'Gibbs':
                        self.samplers[i][variable] = GibbsSampler(info, data.n_individuals)
                    else:
                        self.samplers[i][variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])

    def _initialize_sufficient_statistics(self, data, models, realizations):
        for i, model in enumerate(models):
            suff_stats = model.compute_sufficient_statistics(data, realizations[i])
            self.sufficient_statistics.append(suff_stats)

    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def run(self, models, data, initial_clusters=None):

        # Initialize Model
        self._initialize_seed(self.seed)

        # Initialize first the random variables
        # TODO : Check if needed - model.initialize_random_variables(data)

        if initial_clusters is None:
            # Initialize random clusters
            initial_clusters = np.random.random(size=(data.n_individuals, self.nb_clusters)).T
            initial_clusters = initial_clusters / initial_clusters.sum(axis=0, keepdims=True)

        # Then initialize the Realizations (from the random variables)
        realizations_all = []
        for model in models:
            realizations = model.get_realization_object(data.n_individuals)

            # Smart init the realizations
            realizations_all.append(model.smart_initialization_realizations(data, realizations))

        realizations = realizations_all

        # Initialize Algo
        self._initialize_algo(data, models, realizations, initial_clusters=initial_clusters)

        # Iterate
        for it in range(self.algo_parameters['n_iter']):
            self.iteration(data, models, realizations)
            if self.output_manager is not None:  # TODO better this, should work with nones
                # self.output_manager.iteration(self, data, model, realizations)

                # Temporary fix
                if it % self.output_manager.periodicity_print == 0:
                    print(self)
                    for model in models:
                        print(model)
            self.current_iteration += 1

        print("The standard deviation of the noise at the end of the calibration is ",
              [model.parameters['noise_std'] for model in models])
        return realizations

    def iteration(self, data, models, realizations):
        """
        MCMC-SAEM iteration.
        1. Sample : MC sample successively of the populatin and individual variales
        2. Maximization step : update model parameters from current population/individual variables values.
        :param data:
        :param model:
        :param realizations:
        :return:
        """

        # Computing clusters step
        individual_attachments = torch.zeros((len(models), data.n_individuals))
        for i, model in enumerate(models):
            log_prob = model.compute_individual_attachment_tensorized_mcmc(data, realizations[i])
            for key in realizations[i].reals_ind_variable_names:
                log_prob += self.temperature_inv * model.compute_regularity_realization(realizations[i][key]).sum(dim=1).reshape(data.n_individuals)
            individual_attachments[i] = self.pi[i] * torch.exp(-log_prob)
        individual_attachments = torch.clamp(individual_attachments, 1e-32)
        proba_clusters = individual_attachments / individual_attachments.sum(axis=0, keepdims=True)

        if self.stochastic:
            # Sample from it
            acc = np.cumsum(proba_clusters.detach().numpy(), axis=0).T
            r = np.random.uniform(size=(data.n_individuals))[:, np.newaxis]
            clusters = np.argmax((acc > r).astype(float), axis=1)
            clusters = np.eye(self.nb_clusters)[clusters].T
        else:
            clusters = proba_clusters
        # Update pi
        self.pi = proba_clusters.sum(axis=1) / float(data.n_individuals)

        for i, model in enumerate(models):
            # Sample step
            for key in realizations[i].reals_pop_variable_names:
                self.samplers[i][key].sample(data, model, realizations[i], self.temperature_inv, clusters=clusters[i])
            for key in realizations[i].reals_ind_variable_names:
                self.samplers[i][key].sample(data, model, realizations[i], self.temperature_inv, clusters=clusters[i])

        # Maximization step
        self._maximization_step(data, models, realizations, clusters=clusters)

        for i, model in enumerate(models):
            model.update_MCMC_toolbox(['all'], realizations[i])

        # Update the likelihood with the new noise_var
        # TODO likelihood is computed 2 times, remove this one, and update it in maximization step ?
        # TODO or ar the update of all sufficient statistics ???
        # self.likelihood.update_likelihood(data, model, realizations)

        # Annealing
        if self.algo_parameters['annealing']['do_annealing']:
            self._update_temperature()

    def _update_temperature(self):
        """
        Update the temperature according to an oscillating scheme.
        :return:
        """
        params = self.algo_parameters['annealing']
        b = params['range']
        c = params['delay']
        r = params['period']
        k = self.current_iteration
        kappa = c + float(k)/(2.*r*np.pi)
        self.temperature = max(1. + b * np.sin(kappa)/kappa, 0.1)
        self.temperature_inv = 1./self.temperature

    def _maximization_step(self, data, models, realizations, clusters=None):
        """
         Maximization step as in the EM algorith.
        In practice parameters are set to current realizations (burn-in phase),
        or as a barycenter with previous realizations.
        :param data:
        :param model:
        :param realizations:
        :return:
        """
        burn_in_phase = self._is_burn_in()  # The burn_in is true when the maximization step is memoryless
        for i, model in enumerate(models):
            if burn_in_phase:
                model.update_model_parameters(data, realizations[i], burn_in_phase, clusters=clusters[i])
            else:
                sufficient_statistics = model.compute_sufficient_statistics(data, realizations[i])
                burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter'] + 1)
                self.sufficient_statistics[i] = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                                 for k, v in self.sufficient_statistics[i].items()}
                model.update_model_parameters(data, self.sufficient_statistics[i], burn_in_phase, clusters=clusters[i])

    def _is_burn_in(self):
        """
        Check if current iteration is in burn-in phase.
        :return:
        """
        return self.current_iteration < self.algo_parameters['n_burn_in_iter']

    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Instance of {0} algo \n".format(self.name)
        out += "Iteration {0}\n".format(self.current_iteration)
        out += "=Samplers \n"
        for i, samplers in enumerate(self.samplers):
            out += "\nModel {0}\n".format(i + 1)
            for sampler_name, sampler in samplers.items():
                acceptation_rate = torch.mean(sampler.acceptation_temp.detach()).item()
                out += "    {} rate : {:.2f}%, std: {:.5f}\n".format(sampler_name, 100 * acceptation_rate,
                                                                     sampler.std.mean())

        if self.algo_parameters['annealing']['do_annealing']:
            out += "\nAnnealing \n"
            out += "Temperature : {0}\n".format(self.temperature)
        return out

    #############
    ## HMC
    #############

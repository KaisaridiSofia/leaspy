import torch
import numpy as np

from ..abstract_algo import AbstractAlgo
from ..samplers.gibbs_sampler import GibbsSampler
from ..samplers.old_gibbs_sampler import OldGibbsSampler
from ..samplers.fast_gibbs_sampler import FastGibbsSampler
from ..samplers.metropolis_hastings_sampler import MetropolisHastingsSampler
from ..samplers.hmc_sampler import HMCSampler

from leaspy.models.utils.initialization.model_initialization import initialize_ordinal


class IPMixtureFitMCMC(AbstractAlgo):

    def __init__(self, settings):

        super().__init__()

        # Algorithm parameters
        self.set_output_manager(settings.logs)
        self.algo_parameters = settings.parameters
        self.seed = settings.seed
        self.name = 'IPMixture_MCMC'

        # Realizations and samplers
        self.realizations = None
        self.task = None
        self.samplers = None
        self.sufficient_statistics = None
        self.current_iteration = 0
        self.loss = settings.loss
        self.stochastic = False

        # Annealing
        self.temperature_inv = 1.
        self.temperature = 1.

        self.attachment = None
        self._probas = None

        self.output_function = None
        if 'output_function' in self.algo_parameters:
            self.output_function = self.algo_parameters['output_function']

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations, initial_clusters):
        """
        Initialize the samplers, annealing, MCMC toolbox and sufficient statistics.
        :param data:
        :param model:
        :param realizations:
        :return: realizations
        """
        if model.loss != 'MSE':  # non default loss from model
            assert self.loss in ['MSE', model.loss], \
                f"You provided inconsistent loss: '{model.loss}' for model and '{self.loss}' for algo."
            # set algo loss to the one from model
            self.loss = model.loss
        else:
            # set model loss from algo
            model.loss = self.loss

        # Handling addition of parameters in case of ordinal loss
        if model.loss == 'ordinal':
            # Reinitialize model with added parameters
            initialize_ordinal(model, data, self.algo_parameters)
            realizations = model.get_realization_object(data.n_individuals)

        # Samplers
        self._initialize_samplers(model, data)
        # MCMC toolbox (cache variables for speed-ups + tricks)
        model.initialize_MCMC_toolbox()
        model.loss = self.loss
        self._initialize_sufficient_statistics(data, model, realizations)
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

    def _initialize_samplers(self, model, data):
        """
        Instanciate samplers for Gibbs / HMC sampling as a dictionnary samplers {name: sampler}
        :param model:
        :param data:
        :return:
        """
        infos_variables = model.random_variable_informations()
        self.samplers = dict.fromkeys(infos_variables.keys())
        for variable, info in infos_variables.items():
            if info["type"] == "individual":
                if self.algo_parameters['sampler_ind'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                elif self.algo_parameters['sampler_ind'] == 'OldGibbs':
                    self.samplers[variable] = OldGibbsSampler(info, data.n_individuals)
                elif self.algo_parameters['sampler_ind'] == 'FastGibbs':
                    self.samplers[variable] = FastGibbsSampler(info, data.n_individuals)
                elif self.algo_parameters['sampler_ind'] == 'MH':
                    self.samplers[variable] = MetropolisHastingsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])
            else:
                if self.algo_parameters['sampler_pop'] == 'Gibbs':
                    self.samplers[variable] = GibbsSampler(info, data.n_individuals)
                elif self.algo_parameters['sampler_ind'] == 'OldGibbs':
                    self.samplers[variable] = OldGibbsSampler(info, data.n_individuals)
                elif self.algo_parameters['sampler_ind'] == 'FastGibbs':
                    self.samplers[variable] = FastGibbsSampler(info, data.n_individuals)
                elif self.algo_parameters['sampler_ind'] == 'MH':
                    self.samplers[variable] = MetropolisHastingsSampler(info, data.n_individuals)
                else:
                    self.samplers[variable] = HMCSampler(info, data.n_individuals, self.algo_parameters['eps'])

    def _initialize_sufficient_statistics(self, data, model, realizations):
        suff_stats = model.compute_sufficient_statistics(data, realizations, self._probas)
        self.sufficient_statistics = suff_stats

    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def run(self, model, data, initial_clusters=None):

        # Initialize Model
        self._initialize_seed(self.seed)

        # Initialize first the random variables
        # TODO : Check if needed - model.initialize_random_variables(data)

        if initial_clusters is None:
            # Initialize random clusters
            initial_clusters = torch.exp(torch.normal(size=(data.n_individuals, model.nb_clusters))).T
            initial_clusters = initial_clusters / initial_clusters.sum(dim=0, keepdim=True)

        self._probas = torch.tensor(initial_clusters)

        # Then initialize the Realizations (from the random variables)
        realizations = model.get_realization_object(data.n_individuals)

            # Smart init the realizations
        realizations = model.smart_initialization_realizations(data, realizations)

        self.realizations = realizations

        # Initialize Algo
        self._initialize_algo(data, model, realizations, initial_clusters=initial_clusters)

        # Iterate
        for it in range(self.algo_parameters['n_iter']):
            self.iteration(data, model, realizations)
            if self.output_manager is not None:  # TODO better this, should work with nones
                # self.output_manager.iteration(self, data, model, realizations)

                # Temporary fix
                if it % self.output_manager.periodicity_print == 0:
                    print(self)
                    print(model)
                    if self.output_function is not None:
                        self.output_function(model, self)
            self.current_iteration += 1

        print("The standard deviation of the noise at the end of the calibration is ",
              model.parameters['noise_std'])
        return realizations

    def iteration(self, data, model, realizations):
        """
        MCMC-SAEM iteration.
        1. Sample : MC sample successively of the populatin and individual variales
        2. Maximization step : update model parameters from current population/individual variables values.
        :param data:
        :param model:
        :param realizations:
        :return:
        """
        proba_clusters = self._probas

        self.attachment = None
        for key in realizations.reals_pop_variable_names:
            self.attachment = self.samplers[key].sample(data, model, realizations, self.temperature_inv, previous_attachment=self.attachment)
        for key in realizations.reals_ind_variable_names:
            self.samplers[key].sample(data, model, realizations, self.temperature_inv, model_kwargs={'proba_clusters':proba_clusters}, previous_attachment=None)

        # Computing clusters step
        individual_attachments = torch.zeros((model.nb_clusters, data.n_individuals))
        for i in range(model.nb_clusters):
            individual_attachments[i] -= model.compute_regularity_realization(realizations['tau_xi'], cluster=i).sum(
                dim=1).reshape(data.n_individuals)
        proba_clusters = torch.nn.Softmax(dim=0)(torch.clamp(individual_attachments, -100.))
        self._probas = proba_clusters
        # Maximization step
        self._maximization_step(data, model, realizations, clusters=proba_clusters)

        model.update_MCMC_toolbox(['all'], realizations)

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
        kappa = c + 2. * float(k) * np.pi / r
        self.temperature = max(1. + b * np.sin(kappa)/kappa, 0.1)
        self.temperature_inv = 1./self.temperature

    def _maximization_step(self, data, model, realizations, clusters=None):
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
        if burn_in_phase:
            model.update_model_parameters(data, realizations, burn_in_phase, clusters=clusters)
        else:
            sufficient_statistics = model.compute_sufficient_statistics(data, realizations, clusters)
            burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter'] + 1)
            self.sufficient_statistics = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                             for k, v in self.sufficient_statistics.items()}
            model.update_model_parameters(data, self.sufficient_statistics, burn_in_phase)

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
        for sampler_name, sampler in self.samplers.items():
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

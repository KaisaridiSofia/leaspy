import warnings

from leaspy.algo.algo_factory import AlgoFactory
from leaspy.io.data.dataset import Dataset
#from leaspy.io.outputs.result import Result # not used
from leaspy.io.settings.model_settings import ModelSettings
from leaspy.models.model_factory import ModelFactory
from leaspy.io.logs.visualization.plotting import Plotting


class Leaspy:
    r"""
    Leaspy object class.
    This is the main class of the Leaspy package. It is used to instantiate a Leaspy class object, fit models,
    run algorithms and simulations.

    Attributes
    ----------
    model: leaspy.models.abstract_model.AbstractModel
        The model used for the computation. The available models are:
            - 'logistic' - suppose that every modality follow a logistic curves across time. This model performs a dimensionality reduction of the modalities.

            - 'logistic_parallel' - idem & suppose also that every modality have the same slope at inflexion point

            - 'univariate' - a 'logistic' model for a single modality => do not perform a dimensionality reduction.
    type: str
        Name of the model - must be one of the three listed above.
    plotting: leaspy.utils.output.visualization.plotting.Plotting

    Methods
    -------
    fit(data, algorithm_settings)
        Estimate the model's parameters for a given dataset, a given model and a given algorithm.
        These model's parameters correspond to the fixed-effects of the mixed effect model.
    calibrate(data, algorithm_settings)
        Duplicates of the ``fit`` function.
    personalize(data, settings)
        From a model, estimate individual parameters for each ID of a given dataset.
        These individual parameters correspond to the random-effects of the mixed effect model.
    simulate(individual_parameters, data, settings)
        Generate longitudinal synthetic patients data from a given model, a given collection of individual parameters
        and some given settings.
    estimate(timepoints, individual_parameters)
        Return the value of the features for an individual who is characterized by its individual parameters :math:`z_i`
        at time-points :math:`(t_{i,j})` that can be a unique time-point or a list of time-points.
    load(path_to_model_settings)
        Instantiate a Leaspy object from json model parameter file.
    save(path)
        Save Leaspy object as json model parameter file.
    check_if_initialized()
        Check if model is initialized.
    """

    def __init__(self, model_name, **kwargs):
        """
        Instantiate a Leaspy class object.

        Parameters
        ----------
        model_name : str
            Model's name
        """
        self.model = ModelFactory.model(model_name, **kwargs)
        self.type = model_name
        self.plotting = Plotting(self.model)

    def fit(self, data, algorithm_settings):
        r"""
        Estimate the model's parameters :math:`\theta` for a given dataset, a given model and a given algorithm.
        These model's parameters correspond to the fixed-effects of the mixed effect model.

        Parameters
        ----------
        data: leaspy.io.data.data.Data
            Contains the information of the individuals, in particular the time-points :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        algorithm_settings: leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Contains the algorithm's settings.

        Examples
        --------
        Fit a logistic model on a longitudinal dataset, display the group parameters and plot the
        group average trajectory.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Plotter
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> leaspy_logistic.fit(data, settings)
        >>> print(leaspy_logistic.model.parameters)
        {'g': tensor([-0.4441,  1.9722,  1.6657,  0.1368,  0.8728]),
         'v0': tensor([-3.2442, -3.2942, -3.3763, -2.4901, -3.0032]),
         'betas': tensor([[ 0.0196,  0.0910],
                 [ 0.0559,  0.0291],
                 [-0.0038, -0.1261],
                 [ 0.0988,  0.0767]]),
         'tau_mean': tensor(80.5250),
         'tau_std': tensor(8.1284),
         'xi_mean': 0.0,
         'xi_std': tensor(0.6834),
         'sources_mean': 0.0,
         'sources_std': 1.0,
         'noise_std': tensor(0.0972)}
         >>> leaspy_logistic.plotting.average_trajectory()
        """
        algorithm = AlgoFactory.algo("fit", algorithm_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        if not self.model.is_initialized:
            self.model.initialize(dataset)
        algorithm.run(self.model, dataset)

        # Update plotting
        self.plotting.update_model(self.model)

    def calibrate(self, data, algorithm_settings):
        r"""
        Duplicates of the ``fit`` method. Refer to the ``fit`` documentation.

        Parameters
        ----------
        data: leaspy.io.data.data.Data
            Contains the information of the individuals, in particular the time-points :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        algorithm_settings: leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Contains the algorithm's settings.
        """
        self.fit(data, algorithm_settings)

    def personalize(self, data, settings):
        r"""
        From a model, estimate individual parameters for each `ID` of a given dataset.
        These individual parameters correspond to the random-effects :math:`(z_{i,j})` of the mixed effect model.

        Parameters
        ----------
        data: leaspy.io.data.data.Data
            Contains the information of the individuals, in particular the time-points :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        settings: leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Contains the algorithm's settings.

        Returns
        -------
        leaspy.io.outputs.individual_parameters.IndividualParameters
            Aggregates computed individual parameters and input data.

        Examples
        --------
        Compute the individual parameters for a given longitudinal dataset & display the histogram of the
        log-acceleration.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Plotter
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_parameters = leaspy_logistic.personalize(data, personalize_settings)
        The standard deviation of the noise at the end of the personalization is of 0.0929
        >>> individual_parameters.to_dataframe()
        """
        # Check if model has been initialized
        self.check_if_initialized()

        algorithm = AlgoFactory.algo("personalize", settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        individual_parameters, noise_std = algorithm.run(self.model, dataset)
        return individual_parameters

    def simulate(self, individual_parameters, data, settings):
        r"""
        Generate longitudinal synthetic patients data from a given model, a given collection of individual parameters
        and some given settings.
        This procedure learn the joined distribution of the individual parameters and baseline age of the subjects
        present in ``individual_parameters`` and ``data``respectively to sample new patients from this joined distribution.
        The model is used to compute for each patient their scores from the individual parameters.
        The number of visits per patients is set in ``settings['parameters']['mean_number_of_visits']`` and
        ``settings['parameters']['std_number_of_visits']`` which are set by default to 6 and 3 respectively.

        Parameters
        ----------
        individual_parameters: leaspy.io.outputs.individual_parameters.IndividualParameters
            Contains the individual parameters.
        data: leaspy.io.data.data.Data
            Data object
        settings: leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Contains the algorithm's settings.

        Returns
        -------
        simulated_data: leaspy.io.outputs.result.Result
            Contains the generated individual parameters & the corresponding generated scores.

        Examples
        --------
        Simulate new individual from a given longitudinal, a given model and a given algorithms.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Plotter
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_params = leaspy_logistic.personalize(data, personalize_settings)
        >>> simulated_data = leaspy_logistic.simulate(individual_params, data, AlgorithmSettings('simulation', seed=0))
        """
        # Check if model has been initialized
        self.check_if_initialized()

        algorithm = AlgoFactory.algo("simulate", settings)
        simulated_data = algorithm.run(self.model, individual_parameters, data)
        return simulated_data

    def estimate_old(self, timepoints, individual_parameters):
        r"""
        Return the value of the features for an individual who is characterized by its individual parameters
        :math:`z_i` at time-points :math:`(t_{i,j})` that can be a unique time-point or a list of time-points.
        This functions returns :math:`f(\theta, z_i, (t_{i,j}))`, where :math:`\theta` are the population parameters.
        It is intended to compute reconstructed data, impute missing values and predict future time-points.

        Parameters
        ----------
        timepoints: int, float, list [int or float]
            Corresponds to the time-points to estimate.
        individual_parameters: dict [str, int or float or torch.Tensor]
            Corresponds to the individual parameters of a single individual.

        Returns
        -------
        individual_trajectory: list of torch tensors, each of shape the feature space
            Values of the modality of the individual at the different time-points.

        Examples
        --------
        Estimate the features of the individual at 70, 74 and 80 years old.

        >>> leaspy = Leaspy.load('path/to/model_parameters.json')
        >>> timepoints = [70, 80]
        >>> individual_parameters = { 'xi': 0.3, 'tau': 71, 'sources': [0.2, -0.5] }
        >>> logs = leaspy.estimate(timepoints, individual_parameters)
        """
        warnings.warn("estimate_old() is deprecated; use estimate() instead.", DeprecationWarning)

        # Check if model has been initialized
        self.check_if_initialized()

        # Compute the individual trajectory
        individual_trajectory = self.model.compute_individual_trajectory(timepoints, individual_parameters)
        return individual_trajectory # no squeezing so shape (1, n_tpts, n_features)

    def estimate_multi(self, timepoints, individual_parameters):
        r"""
        Compute for multiple individuals, characterized by their individual parameters :math:`(z_i)`,
        all features values at time-points :math:`(t_{i,j})`.
        This functions returns, for every individual :math:`i`, :math:`f(\theta, z_i, (t_{i,j}))` where :math:`\theta`
        are the population parameters. It is intended to compute reconstructed data, impute missing values and predict
        future time-points.

        Parameters
        ----------
        timepoints: array_like[array_like[numeric]]
            Contains, for each individual, the time-points to estimate.
            The size of this list must match the number of individuals described in individual_parameters
            If an element of this list is an empty list, no estimation will be computed for the corresponding individual,
            resulting in a tensor of size(0, n_features).
        individual_parameters: dict
            Corresponds to the individual parameters of one or more individuals.
            Parameters in it can be tensors or not
            cf. AbstractModel.audit_individual_parameters for some more precision on individual parameters.

        Returns
        -------
        individual_trajectory_gen: generator
            Each element, corresponding to each individual (ordered as is), is a torch tensor
            of shape (number of timepoints to estimate for subject x shape of feature space),
            containing features values at the different timepoints.

        es
        ------
        ValueError
            If any checks fails.

        Examples
        --------
        Given the individual parameters of two subjects, estimate the features of the first
        at 70, 74 and 80 years old and at 71 and 72 years old for the second.

        >>> leaspy = Leaspy.load('path/to/model_parameters.json')
        >>> timepoints = [ (70, 74, 80), (71, 72) ]
        >>> individual_parameters = { 'xi': [0.3, 0.1], 'tau': [71, 59], 'sources': [[0.2, -0.5],[0,0]] }
        >>> logs = leaspy.estimate_multi(timepoints, individual_parameters)
        """
        warnings.warn("estimate_multi() is deprecated; use estimate() instead.", DeprecationWarning)

        # Check if model has been initialized
        self.check_if_initialized()

        # Check individual parameters consistency/compatibility with model
        # and get number of individuals present and generator of tensorized ips
        ips_info = self.model.audit_individual_parameters(individual_parameters)
        n_inds = ips_info['nb_inds']
        t_ips_gen = ips_info['tensorized_ips_gen']

        # Check type and size compatibility of timepoints
        try:
            n_tpts = len(timepoints)
            iter(timepoints)
            # check that each element of timepoints is an iterable
            list(map(lambda set_tpts_i: iter(set_tpts_i), timepoints))
        except TypeError:
            raise ValueError('Timepoints should be a array_like each element containing a set of timepoints ' \
                             'to estimate for corresponding individual.')

        if n_tpts != n_inds:
            raise ValueError('There must be as many sets of timepoints as individuals parametrized. ' \
                             'You gave {} sets of timepoints and {} individuals.'.format(n_tpts, n_inds))

        # Generator of individual trajectories (we skip ips checks and tensorization as were done here)
        traj_gen = (self.model.compute_individual_trajectory(tpts_i, t_ips_i, skip_ips_checks=True).squeeze(0)
                        for tpts_i, t_ips_i in zip(timepoints, t_ips_gen))

        # Return a generator (lazy computation)
        return traj_gen

    def estimate(self, timepoints, individual_parameters):
        r"""
        Description

        Parameters
        ----------
        timepoints: dictionary {string/int: array_like[numeric]
            Contains, for each individual, the time-points to estimate.
        individual_parameters: IndividualParameters object
            Corresponds to the individual parameters of individuals.

        Returns
        -------
        individual_trajectory: dict
            Key: patient indices. Value : Numpy array of the estimated value, in the shape
            (number of timepoints, number of features)

        Examples
        --------
        Given the individual parameters of two subjects, estimate the features of the first
        at 70, 74 and 80 years old and at 71 and 72 years old for the second.

        >>> leaspy = Leaspy.load('path/to/model_parameters.json')
        >>> individual_parameters = IndividualParameters.load('path/to/individual_parameters.json')
        >>> timepoints = { 'index_1': (70, 74, 80), 'index_2': (71, 72) }
        >>> logs = leaspy.estimate_multi(timepoints, individual_parameters)
        """

        estimations = {}

        for index, time in timepoints.items():
            ip = individual_parameters[index]
            est = self.model.compute_individual_trajectory(time, ip)

            estimations[index] = est[0].numpy()

        return estimations

    def check_if_initialized(self):
        """
        Check if model is initialized.

        Raises
        ------
        ValueError
            Raise an error if the model has not been initialized.
        """
        if not self.model.is_initialized:
            raise ValueError("Model has not been initialized")

    @classmethod
    def load(cls, path_to_model_settings):
        """
        Instantiate a Leaspy object from json model parameter file or the corresponding dictionary
        This function can be used to load a pre-trained model.

        Parameters
        ----------
        path_to_model_settings: str of dict
            Path of the model's settings of loaded json in a dictionary

        Returns
        -------
        leaspy.Leaspy
            An instanced Leaspy object with the given population parameters :math:`\theta`.

        Examples
        --------
        Load a pre-trained model.

        >>> from leaspy import Leaspy
        >>> leaspy_univariate = Leaspy.load('outputs/leaspy-univariate_model-seed0.json')
        """
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.initialize_MCMC_toolbox()
        leaspy.model.is_initialized = True

        # Update plotting
        leaspy.plotting.update_model(leaspy.model)

        return leaspy


    def save(self, path):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path: str
            Path to store the model's parameters.

        Examples
        --------

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> leaspy_logistic.fit(data, settings)
        The standard deviation of the noise at the end of the calibration is 0.0726
        >>> leaspy_logistic.save('outputs/leaspy-logistic_model-seed0.json')
        """
        self.check_if_initialized()
        self.model.save(path)

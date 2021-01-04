import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.io.data.data import Data
from leaspy.io.outputs.result import Result


class SimulationAlgorithm(AbstractAlgo):
    r"""
    SimulationAlgorithm object class.
    This algorithm simulate new data given existing one by learning the individual parameters joined distribution.

    You can choose to only learn the distribution of a group of patient. To do so, choose the cofactor and the cofactor
    state of the wanted patient in the settings. Exemple - For Alzheimer patient, you can load a genetic cofactor
    informative of the APOE4 carriers. Choose cofactor 'genetic' and cofactor_state 'APOE4' to simulate only
    APOE4 carriers.

    Attributes
    ----------
    algo_parameters : dict
        Contains the algorithm's parameters.
    bandwidth_method : float or str or callable, optional
        Bandwidth argument used in scipy.stats.gaussian_kde in order to learn the patients' distribution.
    cofactor : str, optional (default = None)
        The cofactor used to select the wanted group of patients (ex - 'genes'). It must correspond to an existing
        cofactor in the attribute `Data` of the input `result` of the ``run`` method. See ``SimulationAlgorithm.run``
        documentation.
    cofactor_state : str, optional (default None)
        The cofactor state used to select the  wanted group of patients (ex - 'APOE4'). It must correspond to an
        existing cofactor state in the attribute `Data` of the input `result` of the ``run`` method. See
        ``SimulationAlgorithm.run`` documentation.
    features_bounds : bool or dict : [`str`, `float`, `float`], (default False)
        Specify if the scores of the generated subjects must be bounded. This parameter can express in two way:
        - `bool` : the bounds are the maximum and minimum scores observed in the ``results.data`` object.
        - `dict` : the user has to set the min and max bounds for every features. For example:
                    ``{'feature1': (score_min, score_max), 'feature2': (score_min, score_max), ...}``
    mean_number_of_visits : int (default 6)
        Average number of visits of the simulated patients.
        Examples - choose 5 => in average, a simulated patient will have 5 visits.
    name : ``"simulation"``
        Algorithm's name.
    noise : str or float or list or None, (default "default")
        Wanted level of gaussian noise in the generated scores.
            - Set to "default", the noise added to each feature score correspond to the reconstruction error for each
            feature.
            - Set noise to ``None`` will lead to patients having "perfect progression" of their scores, i.e.
            following exactly a logistic curve.
            - Set a float will add for each feature's scores a noise of standard deviation the given float.
            - Set a iterable will add for the feature j a noise of standard deviation noise[j]. Then, noise must be
            of length the number of features' scores.
    number_of_subjects : int
        Number of subject to simulate.
    reparametrized_age_bounds : list [float], optional (default None), length = 2
        Set the minimum and maximum age of the generated reparametrized subjects' ages. See Notes section.
        Example - reparametrized_age_bounds = (65, 70)
    seed : int
        Used by numpy.random & torch.random for reproducibility.
    sources_method : str, {'full_kde', 'normal_sources'}
        - ``"full_kde"`` : the sources are also learned with the gaussian kernel density estimation.
        - ``"normal_sources"`` : the sources are generated as multivariate normal distribution linked with the other
        individual parameters.
    std_number_of_visits : int
        Standard deviation used into the generation of the number of visits per simulated patient.

    Methods
    -------
    run(model, individual_parameters, data)
        Run the simulation of new patients for some given leaspy model, individual parameters and data.

    Notes
    -----
    One can choose to set the interval of the reparametrized baseline age of the simulated subjects. By doing so, the
    baseline age are no more jointly learned with individual parameters. Instead, the baseline ages are derived from
    the simulated individual parameters and the reparametrized baseline age which is sample from an uniform distribution
    in the set interval.

    By definition, the relation between age and reparametrized age is:

    .. math:: \psi_i (t) = e^{\xi_i} (t - \tau_i) + \bar{\tau}

    with :math:`t` the real age, :math:`\psi_i (t)` the reparametrized age, :math:`\xi_i` the individual
    log-acceleration parameter, :math:`\tau_i` the individual time-shift parameter and :math:`\bar{\tau}` the mean
    conversion age derivated by the `model` object.
    """

    def __init__(self, settings):
        """
        Process initializer function that is called by Leaspy().simulate.

        Parameters
        ----------
        settings : leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Set the class attributes.

        Raises
        ------
        ValueError
            If ``settings.parameters['sources_method']`` is not one of the two option allowed -
            "full_kde" or "normal_sources".
        TypeError
            If the type of ``settings.parameters['features_bounds']`` is not `bool` or `dict`.
        """
        super().__init__()

        # TODO: put it in abstract_algo + add settings=None in AbstractAlgo __init__ method
        self.algo_parameters = settings.parameters
        self.name = settings.name
        self.seed = settings.seed

        self._initialize_seed(self.seed)

        self.bandwidth_method = settings.parameters['bandwidth_method']
        self.cofactor = settings.parameters['cofactor']
        # TODO: check that the loaded cofactors are converted into strings!
        self.cofactor_state = settings.parameters['cofactor_state']
        self.features_bounds = settings.parameters['features_bounds']
        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.noise = settings.parameters['noise']
        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.reparametrized_age_bounds = settings.parameters['reparametrized_age_bounds']
        self.sources_method = settings.parameters['sources_method']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']

        if self.sources_method not in ("full_kde", "normal_sources"):
            raise ValueError('The "sources_method" parameter must be "full_kde" or "normal_sources"!')

        if type(self.features_bounds) not in [bool, dict]:
            raise TypeError('The type of the "features_bounds" parameter must be %s or %s, not %s!'
                            % (str(bool), str(dict), str(type(self.features_bounds))))

        if self.reparametrized_age_bounds and (len(self.reparametrized_age_bounds) != 2):
            raise ValueError("The parameter 'reparametrized_age_bounds' must contain exactly two elements, "
                             "its lower bound and its upper bound. You gave {0}".format(self.reparametrized_age_bounds))

    def _check_cofactors(self, data):
        """
        Check the value.

        Parameters
        ----------
        data : leaspy.io.data.data.Data
            Contains the cofactors and cofactors' states.

        Raises
        ------
        ValueError
            Raised if the parameters "cofactor" and "cofactor_state" do not receive a valid value.
        """
        def reformat_str(string, replace=True):
            result = string.replace('[', "").replace(']', "")
            if replace:
                result = result.replace(',', " or")
            return result

        cofactors = {}
        for ind in data.individuals.values():
            if bool(ind.cofactors):
                for key, val in ind.cofactors.items():
                    if key in cofactors.keys():
                        cofactors[key] += [val]
                    else:
                        cofactors[key] = [val]
        for key, val in cofactors.items():
            cofactors[key] = np.unique(val)

        if self.cofactor not in cofactors.keys():
            raise ValueError('The input "cofactor" parameter %s does not correspond to any cofactor in your data! '
                             'The available cofactor(s) are %s.'
                             % (self.cofactor, reformat_str(str(list(cofactors.keys())))))
        if self.cofactor_state not in cofactors[self.cofactor]:
            raise ValueError('The input "cofactor_state" parameter "%s" does not correspond to any cofactor state'
                             ' in your data! The available cofactor states for "%s" are %s.'
                             % (self.cofactor_state, self.cofactor, reformat_str(str(cofactors[self.cofactor]))))

    @staticmethod
    def _get_mean_and_covariance_matrix(m):
        """
        Compute the empirical mean and covariance matrix of the input. Twice faster than `numpy.cov`.

        Parameters
        ----------
        m : torch.Tensor, shape = (n_individual_parameters, n_subjects)
            Input matrix - one row per individual parameter distribution (xi, tau etc).

        Returns
        -------
        mean : torch.Tensor
            Mean by variable, shape = (n_individual_parameters,).
        covariance :  torch.Tensor
            Covariance matrix, shape = (n_individual_parameters, n_individual_parameters).
        """
        m_exp = torch.mean(m, dim=0)
        x = m - m_exp[None, :]
        cov = 1 / (x.size(0) - 1) * x.t() @ x
        return m_exp, cov

    @staticmethod
    def _sample_sources(bl, tau, xi, source_dimension, df_mean, df_cov):
        """
        Simulate individual sources given baseline age bl, time-shift tau, log-acceleration xi & sources dimension.

        Parameters
        ----------
        bl : float
            Baseline age of the simulated patient.
        tau : float
            Time-shift of the simulated patient.
        xi : float
            Log-acceleration of the simulated patient.
        source_dimension : int
            Sources' dimension of the simulated patient.
        df_mean : torch.Tensor, shape = (n_individual_parameters,)
            Mean values per individual parameter type (bl_mean, tau_mean, xi_mean & sources_means) (1-dimensional).
        df_cov : torch.Tensor, shape = (n_individual_parameters, n_individual_parameters)
            Empirical covariance matrix of the individual parameters (2-dimensional).

        Returns
        -------
        torch.Tensor
            Sources of the simulated patient, shape = (n_sources, ).
        """
        x_1 = torch.tensor([bl, tau, xi], dtype=torch.float32)

        mu_1 = df_mean[:3].clone()
        mu_2 = df_mean[3:].clone()

        sigma_11 = df_cov.narrow(0, 0, 3).narrow(1, 0, 3).clone()
        sigma_22 = df_cov.narrow(0, 3, source_dimension).narrow(1, 3, source_dimension).clone()
        sigma_12 = df_cov.narrow(0, 3, source_dimension).narrow(1, 0, 3).clone()

        mean_cond = mu_2 + sigma_12 @ sigma_11.inverse() @ (x_1 - mu_1)
        cov_cond = sigma_22 - sigma_12 @ sigma_11.inverse() @ sigma_12.transpose(0, -1)

        return torch.distributions.multivariate_normal.MultivariateNormal(mean_cond, cov_cond).sample()

    def _get_number_of_visits(self):
        """
        Simulate number of visits for a new simulated patient based of attributes 'mean_number_of_visits' &
        'std_number_of_visits'.

        Returns
        -------
        number_of_visits : int
            Number of visits.
        """
        # Generate a number of visit around the mean_number_of_visits
        number_of_visits = int(self.mean_number_of_visits)
        if self.mean_number_of_visits != 0:
            number_of_visits += int(torch.normal(torch.tensor(0., dtype=torch.float32),
                                                 torch.tensor(self.std_number_of_visits, dtype=torch.float32)).item())
        return number_of_visits

    def _get_features_bounds(self, results_object):
        """
        Get the bound of the baseline scores of the generated patients. Each generated patient whose baseline is outside
        these bounds are discarded.

        Parameters
        ----------
        results_object : leaspy.io.outputs.result.Result

        Returns
        -------
        features_min : numpy.ndarray
            Lowest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.
        features_max : numpy.ndarray
            Highest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.
        """
        features_min = np.zeros(len(results_object.data.headers))
        features_max = np.ones(len(results_object.data.headers))
        if type(self.features_bounds) is dict:
            assert results_object.data.headers == list(self.features_bounds.keys()), \
                'The keys of your input "features_bounds" do not match the headers of your data!' \
                + '\nThe data headers - %s' % str(results_object.data.headers) \
                + '\nYour "features_bounds" input - %s' % str(list(self.features_bounds.keys()))
            for i, key in enumerate(results_object.data.headers):
                features_min[i] = self.features_bounds[key][0]
                features_max[i] = self.features_bounds[key][1]
            return features_min, features_max
        else:
            df_scores = results_object.data.to_dataframe().groupby('ID').first()
            return df_scores.iloc[:, 1:].min().values, df_scores.iloc[:, 1:].max().values

    def _get_timepoints(self, bl):
        """
        Generate the time points of a subject given his baseline age.

        Parameters
        ----------
        bl : float
            The subject's baseline age.

        Returns
        -------
        ages : `list` [`float`]
            Contains the subject's time points.
        """
        number_of_visits = self._get_number_of_visits()
        if number_of_visits == 1:
            ages = [bl]
        elif number_of_visits == 2:
            ages = [bl, bl + 0.5]
        else:
            ages = [bl, bl + 0.5] + [bl + j for j in range(1, number_of_visits - 1)]
        return ages

    def _get_noise_generator(self, model, results):
        """
        Compute the level of L2 error per feature and return a noise generator or size n_features.

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel.
        results : leaspy.io.outputs.result.Result
            Object containing the computed individual parameters.

        Returns
        -------
        torch.distributions.Normal or None
            A gaussian noise generator. If self.noise is None, the function returns None.

        Raises
        ------
        ValueError
            If the attribute self.noise is an iterable of float of a length different than the number of features.
        """
        if self.noise:
            if self.noise == "default":
                noise = results.get_error_distribution_dataframe(model)
                noise = torch.from_numpy(noise[results.data.headers].values)
                noise *= noise
                noise = torch.sqrt(noise.mean(dim=0))
            else:
                if hasattr(self.noise, '__len__'):
                    if len(self.noise) != len(results.data.headers):
                        raise ValueError("The attribute 'noise' you gave is {}. If you want to specify the level of"
                                         " noise for each feature score, you must give an iterable object of size "
                                         "the number of features, here {}.".format(self.noise,
                                                                                   len(results.data.headers)))
                noise = torch.tensor(self.noise)
            return torch.distributions.Normal(loc=0, scale=noise)

    @staticmethod
    def _get_reparametrized_age(timepoints, tau, xi, tau_mean):
        """
        Returns the subjects' reparametrized ages.

        Parameters
        ----------
        timepoints : np.ndarray, shape = (n_subjects,)
            Real ages of the subjects.
        tau : np.ndarray, shape = (n_subjects,)
            Individual time-shifts.
        xi : np.ndarray, shape = (n_subjects,)
            Individual log-acceleration.
        tau_mean : float
            The mean conversion age derivated by the model.

        Returns
        -------
        np.ndarray, shape = (n_subjects,)
        """
        return np.exp(xi) * (timepoints - tau) + tau_mean

    @staticmethod
    def _get_real_age(repam_ages, tau, xi, tau_mean):
        """
        Returns the subjects' real ages.

        Parameters
        ----------
        repam_ages : np.ndarray, shape = (n_subjects,)
            Reparametrized ages of the subjects.
        tau : np.ndarray, shape = (n_subjects,)
            Individual time-shifts.
        xi : np.ndarray, shape = (n_subjects,)
            Individual log-acceleration.
        tau_mean : float
            The mean conversion age derivated by the model.

        Returns
        -------
        np.ndarray, shape = (n_subjects,)
        """
        return np.exp(-xi) * (repam_ages - tau_mean) + tau

    def _simulate_individual_parameters(self, model, number_of_simulated_subjects, kernel, ss, get_sources,
                                        df_mean, df_cov):
        """
        Compute the simulated individual parameters and timepoints.

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            A subclass object of leaspy AbstractModel.
        number_of_simulated_subjects : int
        kernel : scipy.stats.gaussian_kde
        ss : sklearn.preprocessing.StandardScaler
        get_sources : bool
        df_mean : torch.Tensor, shape = (n_individual_parameters,)
            Mean values per individual parameter type.
        df_cov : torch.Tensor, shape = (n_individual_parameters, n_individual_parameters)
            Empirical covariance matrix of the individual parameters.

        Returns
        -------
        simulated_parameters : dict [str, numpy.ndarray]
            Contains the simulated parameters.
        timepoints : `list` [float]
            Contains the ages of the subjects for all their visits - 2D list with one row per simulated subject.
        """
        samples = kernel.resample(number_of_simulated_subjects).T
        samples = ss.inverse_transform(samples)  # A np.ndarray of shape (n_subjects, n_features)

        # Transform reparametrized baseline age into baseline real age
        samples[:, 0] = self._get_real_age(repam_ages=samples[:, 0],
                                           tau=samples[:, 1],
                                           xi=samples[:, 2],
                                           tau_mean=model.parameters['tau_mean'].item())

        timepoints = list(map(self._get_timepoints, samples[:, 0]))
        # timempoints is a 2D list - one row per simulated subject

        simulated_parameters = {'tau': samples[:, 1], 'xi': samples[:, 2]}
        # xi & tau are 1D array - one value per simulated subject
        if get_sources:
            if self.sources_method == "full_kde":
                simulated_parameters['sources'] = samples[:, 3:]
            elif self.sources_method == "normal_sources":
                # Generate sources
                def simulate_sources(x: np.ndarray) -> np.ndarray:
                    return self._sample_sources(x[0], x[1], x[2], model.source_dimension, df_mean, df_cov).numpy()

                simulated_parameters['sources'] = np.apply_along_axis(simulate_sources, axis=1, arr=samples)
                # sources is np.ndarray of shape (n_subjects, n_sources)

        return simulated_parameters, timepoints

    @staticmethod
    def _simulate_subjects(simulated_parameters, timepoints, model, noise_generator):
        """
        Compute the simulated scores given the simulated individual parameters, timepoints & noise generator.

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            A subclass object of leaspy AbstractModel.
        simulated_parameters : dict [str, numpy.ndarray]
            Contains the simulated parameters.
        timepoints : `list` [float]
            Contains the ages of the subjects for all their visits - 2D list with one row per simulated subject.
        noise_generator : torch.distributions.Normal or None
            A gaussian noise generator. If self.noise is None, the features' score are exactly the ones derived from
            the individual parameters by the model.

        Returns
        -------
        features_values : `list` [numpy.ndarray]
            Contains the scores of all the subjects for all their visits. Contains one entry per subject, each of
            them is a 2D-numpy.ndarray of shape n_visits x n_features.
        """
        features_values = []
        # TODO : parallelize this for loop
        for i in range(len(timepoints)):
            indiv_param = {key: val[i] for key, val in simulated_parameters.items()}
            if 'sources' in indiv_param:
                indiv_param['sources'] = indiv_param['sources'].tolist()
            observations = model.compute_individual_trajectory(timepoints[i], indiv_param)
            # Add the desired noise
            if noise_generator:
                observations += noise_generator.sample([observations.shape[0]])
                observations = observations.clamp(0, 1)

            observations = observations.squeeze(0).detach().numpy()
            features_values.append(observations)

        return features_values

    @staticmethod
    def _get_bounded_subject(features_values, features_min, features_max):
        """
        Select the subject whose scores are within the features boundaries.

        Parameters
        ----------
        features_values : `list` [np.ndarray]
            Contains the scores of all the subjects of all their visits. Each element correspond to a simulated
            subject, these elements are of shape n_vists x n_features.
        features_min : np.ndarray
            Lowest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.
        features_max : np.ndarray
            Highest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.

        Returns
        -------
        indices_of_accepted_simulated_subjects : `list` [float]
        `list` [np.ndarray]
            Contains the scores of all the subjects whose scores are within the features boundaries.
        """

        def _test_subject(bl_score: float, features_min: np.array, features_max: np.array) -> bool:
            return all(features_min <= bl_score) & all(bl_score <= features_max)

        baseline_scores = np.array([scores[0] for scores in features_values])
        indices_of_accepted_simulated_subjects = [i for i, bl_score in enumerate(baseline_scores)
                                                  if _test_subject(bl_score, features_min, features_max)]
        return indices_of_accepted_simulated_subjects, [val for i, val in enumerate(features_values)
                                                        if i in indices_of_accepted_simulated_subjects]

    def run(self, model, individual_parameters, data):
        """
        Run simulation - learn joined distribution of patients' individual parameters and return a results object
        containing the simulated individual parameters and the simulated scores.

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            Subclass object of AbstractModel. Model used to compute the population & individual parameters.
            It contains the population parameters.
        individual_parameters : leaspy.io.outputs.individual_parameters.IndividualParameters
            Object containing the computed individual parameters.

        Notes
        -----
        In simulation_settings, one can specify in the parameters the cofactor & cofactor_state. By doing so,
        one can simulate based only on the subject for the given cofactor & cofactor's state.

        By default, all the subject in results.data are used to estimate the joined distribution.

        Returns
        -------
        leaspy.io.outputs.result.Result
            Contains the simulated individual parameters & individual scores.
        """

        _, dict_pytorch = individual_parameters.to_pytorch()
        results = Result(data, dict_pytorch)

        if self.cofactor is not None:
            self._check_cofactors(data)

        # --------- Get individual parameters & reparametrized baseline ages - for joined density estimation
        # Get individual parameters (optional - & the cofactor states)
        df_ind_param = results.get_dataframe_individual_parameters(cofactors=self.cofactor)
        if self.cofactor_state:
            # Select only subjects with the given cofactor state
            df_ind_param = df_ind_param[df_ind_param[self.cofactor] == self.cofactor_state]
            # Remove the cofactor column
            df_ind_param = df_ind_param.loc[:, df_ind_param.columns != self.cofactor_state]
        # Add the baseline ages
        df_ind_param = results.data.to_dataframe().groupby('ID').first()[['TIME']].join(df_ind_param, how='right')
        # At this point, df_ind_param.columns = ['TIME', 'tau', 'xi', 'sources_0', 'sources_1', ..., 'sources_n']
        distribution = df_ind_param.values
        # Transform baseline age into reparametrized baseline age
        distribution[:, 0] = self._get_reparametrized_age(timepoints=distribution[:, 0],
                                                          tau=distribution[:, 1],
                                                          xi=distribution[:, 2],
                                                          tau_mean=model.parameters['tau_mean'].item())
        # If constraints on baseline reparametrized age have been set
        # Select only the subjects who satisfy the constraints
        if self.reparametrized_age_bounds:
            distribution = np.array([ind for ind in distribution if
                                     min(self.reparametrized_age_bounds) < ind[0] < max(self.reparametrized_age_bounds)])
        # Get sources according the selected sources_method
        get_sources = (model.name != 'univariate')
        if get_sources & (self.sources_method == "normal_sources"):
            # Sources are not learned with a kernel density estimator
            distribution = distribution[:, :3]
            # Get mean by variable & covariance matrix
            # Needed to sample new sources from simulated bl, tau & xi
            df_mean, df_cov = self._get_mean_and_covariance_matrix(torch.from_numpy(df_ind_param.values))
        else:
            df_mean, df_cov = None, None

        # --------- Get joined density estimation of repam bl, tau, xi (and sources if the model is not univariate)
        # Normalize by variable then transpose to learn the joined distribution
        ss = StandardScaler()
        # fit_transform receive an numpy array of shape (n_samples, n_features)
        distribution = ss.fit_transform(distribution).T
        # gaussian_kde receive an numpy array of shape (n_features, n_samples)
        kernel = stats.gaussian_kde(distribution, bw_method=self.bandwidth_method)

        # --------- Simulate new subjects - individual parameters, timepoints and features' scores
        if self.features_bounds:
            number_of_simulated_subjects = 10 * self.number_of_subjects
            # Simulate more subject in order to have enough of them after filtering in order to respect the bounds
        else:
            number_of_simulated_subjects = self.number_of_subjects

        simulated_parameters, timepoints = self._simulate_individual_parameters(
            model, number_of_simulated_subjects, kernel, ss, get_sources, df_mean, df_cov)

        noise_generator = self._get_noise_generator(model, results)

        features_values = self._simulate_subjects(simulated_parameters, timepoints, model, noise_generator)

        # --------- If one wants to select generated subjects based on their baseline scores
        if self.features_bounds:
            # Handle bounds on the generated features
            features_min, features_max = self._get_features_bounds(results)
            #  Test the boundary conditions & filter subjects with features' scores outside the bounds.
            indices_of_accepted_simulated_subjects, features_values = self._get_bounded_subject(
                features_values, features_min, features_max)
            for key, val in simulated_parameters.items():
                simulated_parameters[key] = val[indices_of_accepted_simulated_subjects]

            timepoints = [v for i, v in enumerate(timepoints) if i in indices_of_accepted_simulated_subjects]

            # If too much subjects have been discarded
            while len(features_values) < self.number_of_subjects:
                # Complete to attain the goal
                number_of_simulated_subjects *= self.number_of_subjects / len(indices_of_accepted_simulated_subjects)

                simulated_parameters_bis, timepoints_bis = self._simulate_individual_parameters(
                    model, number_of_simulated_subjects, kernel, ss, get_sources, df_mean, df_cov)

                features_values_bis = self._simulate_subjects(simulated_parameters_bis, timepoints_bis,
                                                              model, noise_generator)

                #  Test the boundary conditions
                indices_of_accepted_simulated_subjects_bis, features_values_bis = self._get_bounded_subject(
                    features_values_bis, features_min, features_max)
                for key, val in simulated_parameters_bis.items():
                    simulated_parameters_bis[key] = val[indices_of_accepted_simulated_subjects_bis]
                timepoints_bis = [v for i, v in enumerate(timepoints_bis)
                                  if i in indices_of_accepted_simulated_subjects_bis]

                # Concatenate with previous generated subjects
                features_values += features_values_bis
                timepoints += timepoints_bis
                for key in simulated_parameters:
                    simulated_parameters[key] = np.concatenate(simulated_parameters[key],
                                                               simulated_parameters_bis[key])

        # --------- Take only the `number_of_simulated_subjects` first generated subjects
        n = self.number_of_subjects
        timepoints = timepoints[:n]
        for key, val in simulated_parameters.items():
            if key in ['tau', 'xi']:
                simulated_parameters[key] = torch.from_numpy(val).view(-1, 1)[:n]
            if key == 'sources':
                simulated_parameters[key] = torch.from_numpy(val)[:n]

        # --------- Give results
        indices = ['Generated_subject_' + '0' * (len(str(n)) - len(str(i))) + str(i) for i in range(1, n + 1)]
        # Ex - for 10 subjects, indices = ["Generated_subject_01", "Generated_subject_02", ..., "Generated_subject_10"]

        simulated_scores = Data.from_individuals(indices=indices,
                                                 timepoints=timepoints,
                                                 values=features_values,
                                                 headers=results.data.headers)
        return Result(data=simulated_scores,
                      individual_parameters=simulated_parameters,
                      noise_std=self.noise)

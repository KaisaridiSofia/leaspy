import math
import time

from ..abstract_algo import AbstractAlgo


class AbstractPersonalizeAlgo(AbstractAlgo):
    """
    Abstract class for "personalize" algorithm.
    Estimation of individual parameters of a given "Data" file with
    a freezed model (already estimated, or loaded from known parameters)

    Attributes
    ----------
    algo_parameters: str
        algorithm's name
     name: str
        algorithm's name
     seed: int
        algorithm's seed (default None)

    Methods
    -------
    _get_individual_parameters(model, data)
        Estimate individual parameters from a data using leaspy object class model & Dataset
    run(model, data)
        Main personalize function, wraps the _get_individual_parameters
    """

    def __init__(self, settings):
        """
        Initialize class object from settings object

        Parameters
        ----------
        settings : leaspy.io.settings.algorithm_settings.AlgorithmSettings
            Settings of the algorithm
        """
        super().__init__()

        # Algorithm parameters
        self.algo_parameters = settings.parameters

        # Name
        self.name = settings.name

        # Seed
        self.seed = settings.seed
        self.loss = settings.loss

    def run(self, model, data):
        """
        Main personalize function, wraps the _get_individual_parameters

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            A subclass object of leaspy AbstractModel.
        data : leaspy.io.data.dataset.Dataset
            Dataset object build with leaspy class objects Data, algo & model

        Returns
        -------
        dict, float
            dict - individual parameters
                exemple {'xi': <list of float>, 'tau': <list of float>, 'sources': <list of list of float>}
            float - estimated noise
                = ( 1 / nber_visits * nbre_dimension * Sum_patient (modelization_scores - real_values) ** 2 ) ** 1/2
        """

        # Set seed
        self._initialize_seed(self.seed)

        # Init the run
        #print("Beginning personalization : std error of the model is {0}".format(model.parameters['noise_std']))
        time_beginning = time.time()

        # Give the model the adequate loss
        model.loss = self.loss

        # Estimate individual parametersabstr
        individual_parameters = self._get_individual_parameters(model, data)

        # Compute the noise with the estimated individual parameters
        _, dict_pytorch = individual_parameters.to_pytorch()
        if self.loss == "MSE":
            squared_diff = model.compute_sum_squared_tensorized(data, dict_pytorch).sum()
            noise_std = math.sqrt(squared_diff.detach().item() / data.n_observations)
        else:
            noise_std = "TODO compute_attachment for other losses"

        # Print run infos
        time_end = time.time()
        diff_time = (time_end - time_beginning)  # / 1000 TODO: why divided by 1000?
        #print("The standard deviation of the noise at the end of the personalization is of {:.4f}".format(noise_std))
        #print("Personalization {1} took : {0}s".format(diff_time, self.name))

        # Transform individual parameters to dictinnary ID / variable_ind
        # indices = data.indices
        # new_individual_parameters = dict.fromkeys(indices)
        # for i, idx in enumerate(indices):
        #    new_individual_parameters[idx] = {}
        #    for j, variable_ind_name in enumerate(model.get_individual_variable_name()):
        #        new_individual_parameters[idx][variable_ind_name] = individual_parameters[j][i].detach().tolist()

        return individual_parameters, noise_std

    def _get_individual_parameters(self, model, data):
        """
        Estimate individual parameters from a data using leaspy object class model & Dataset

        Parameters
        ----------
        model : leaspy.models.abstract_model.AbstractModel
            A subclass object of leaspy AbstractModel.
        data : leaspy.io.data.dataset.Dataset
            Dataset object build with leaspy class objects Data, algo & model

        Raises
        ------
        NotImplementedError
            Method only implemented in child class of AbstractPersonalizeAlgo
        """

        raise NotImplementedError('This algorithm does not present a personalization procedure')

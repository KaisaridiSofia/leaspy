import json
import os
import warnings

from leaspy.io.settings import default_data_dir
from leaspy.io.settings.outputs_settings import OutputsSettings


class AlgorithmSettings:
    """
    Read a algo_parameters json file and create the corresponding algo
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.parameters = None
        self.seed = None
        self.initialization_method = None
        self.loss = None
        self.logs = None

        if name in ['mcmc_saem', 'scipy_minimize', 'simulation', 'mean_real', 'gradient_descent_personalize',
                    'mode_real']:
            self._load_default_values(os.path.join(default_data_dir, 'default_' + name + '.json'))
        else:
            raise ValueError('The algorithm name >>>{0}<<< you provided does not exist'.format(name))
        self._manage_kwargs(kwargs)

    @classmethod
    def load(cls, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        if 'name' not in settings.keys():
            raise ValueError("Your json file must contain a 'name' attribute!")
        algorithm_settings = cls(settings['name'])

        if 'parameters' in settings.keys():
            print("You overwrote the algorithm default parameters")
            algorithm_settings.parameters = cls._get_parameters(settings)

        if 'seed' in settings.keys():
            print("You overwrote the algorithm default seed")
            algorithm_settings.seed = cls._get_seed(settings)

        if 'initialization_method' in settings.keys():
            print("You overwrote the algorithm default initialization method")
            algorithm_settings.initialization_method = cls._get_initialization_method(settings)

        if 'loss' in settings.keys():
            print('You overwrote the algorithm default loss')
        algorithm_settings.loss = cls._get_loss(settings)

        return algorithm_settings

    def _manage_kwargs(self, kwargs):
        if 'seed' in kwargs.keys():
            self.seed = self._get_seed(kwargs)
        if 'initialization_method' in kwargs.keys():
            self.initialization_method = self._get_initialization_method(kwargs)
        if 'loss' in kwargs.keys():
            self.loss = self._get_loss(kwargs)

        for k, v in kwargs.items():
            if k in ['seed', 'initialization_method', 'loss']:
                continue

            if k in self.parameters.keys():
                self.parameters[k] = v
            else:
                self.parameters[k] = v
                warning_message = "The parameter key : >>>{0}<<< you provided is unknown".format(k)
                warnings.warn(warning_message)

        if self.name == 'mcmc_saem':
            if 'n_iter' in kwargs.keys() and ('n_burn_in_iter' not in kwargs.keys() or kwargs['n_burn_in_iter'] is None):
                self.parameters['n_burn_in_iter'] = int(0.9 * kwargs['n_iter'])

            # TODO : For Raphael : what does it mean? Because there are already default value.
            # TODO : Thus, either default value for annealing/iter in the json, either here. Not both.
            if 'n_iter' in kwargs.keys() and 'annealing' not in kwargs.keys():
                self.parameters['annealing']["n_iter"] = int(0.5 * kwargs['n_iter'])

    def _load_default_values(self, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        AlgorithmSettings._check_default_settings(settings)
        self.name = self._get_name(settings)
        self.parameters = self._get_parameters(settings)
        self.seed = self._get_seed(settings)
        self.loss = self._get_loss(settings)

    def set_logs(self, path, **kwargs):
        settings = {
            'path': path,
            'console_print_periodicity': 50,
            'plot_periodicity': 100,
            'save_periodicity': 50
        }

        for k, v in kwargs.items():
            if k in ['console_print_periodicity', 'plot_periodicity', 'save_periodicity']:
                # Todo : Ca devrait planter si v n'est pas un int!!
                settings[k] = v
            else:
                warnings.warn("The kwargs {} you provided is not correct".format(v))

        self.logs = OutputsSettings(settings)

    @staticmethod
    def _check_default_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'name\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'seed' not in settings.keys():
            raise ValueError('The \'settings\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'initialization_method' not in settings.keys():
            raise ValueError(
                'The \'initialization_method\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'loss' not in settings.keys():
            warnings.warn("The \'loss\' key is missing in the algorithm settings (JSON file) you are loading. \
            Its value will be \'MSE\' by default")

    @staticmethod
    def _get_name(settings):
        return settings['name'].lower()

    @staticmethod
    def _get_parameters(settings):
        return settings['parameters']

    @staticmethod
    def _get_seed(settings):
        if settings['seed'] is None:
            return None
        try:
            return int(settings['seed'])
        except ValueError:
            print("The \'seed\' parameter you provided cannot be converted to int")

    @staticmethod
    def _get_initialization_method(settings):
        if settings['initialization_method'] is None:
            return None
        # TODO : There should be a list of possible initialization method. It can also be discussed depending on the algorithms name
        return settings['initialization_method']

    @staticmethod
    def _get_loss(settings):
        if not 'loss' in settings:
            # Return default value for the loss (Mean Squared Error)
            return 'MSE'
        if settings['loss'].lower() in ['mse', 'crossentropy', 'ordinal']:
            return settings['loss']
        else:
            raise ValueError("The loss provided is not recognised. Should be one of [\'MSE\', \'crossentropy\', \'ordinal\']")

    def save(self, path):

        json_settings = {
            "name": self.name,
            "seed": self.seed,
            "parameters" : self.parameters,
            "initialization_method" : self.initialization_method,
            "loss" : self.loss,
            "logs" : self.logs
        }

        with open(os.path.join(path), "w") as json_file:
            json.dump(json_settings, json_file)



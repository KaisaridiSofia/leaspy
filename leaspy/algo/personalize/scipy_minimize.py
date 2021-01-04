from scipy.optimize import minimize
import torch

from .abstract_personalize_algo import AbstractPersonalizeAlgo
from ...io.outputs.individual_parameters import IndividualParameters


class ScipyMinimize(AbstractPersonalizeAlgo):

    def __init__(self, settings):
        super(ScipyMinimize, self).__init__(settings)

        self.verbose = False
        if "verbose" is self.algo_parameters:
            self.verbose = self.algo_parameters["verbose"]

        self.minimize_kwargs = {
            'method': "Powell",
            'options': {
                'xtol': 1e-4,
                'ftol': 1e-4
            },
            # 'tol': 1e-6
        }

        if self.algo_parameters['use_jacobian']:
            self.minimize_kwargs = {
                'method': "BFGS",
                'options': {
                    'gtol': 0.01,
                },
                # 'tol': 1e-6
                'tol':5e-5
            }

    def _set_model_name(self, name):
        """
        Set name attribute.

        Parameters
        ----------
        name: `str`
            Model's name.
        """
        self.model_name = name

    def _initialize_parameters(self, model):
        """
        Initialize individual parameters of one patient with group average parameter.

        Parameters
        ----------
        model: leaspy model class object

        Returns
        -------
        x: `list` [`float`]
            The individual parameters.
            By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)
        """
        x = [model.parameters["xi_mean"], model.parameters["tau_mean"]]
        if model.name != "univariate":
            x += [torch.tensor([0.]) for _ in range(model.source_dimension)]
        return x

    def _get_attachment(self, model, times, values, x):
        """
        Compute model values minus real values of a patient for a given model, timepoints, real values &
        individual parameters.

        Parameters
        ----------
        model: Leaspy model class object
            Model used to compute the group average parameters.
        times: `torch.Tensor`
            Contains the individual ages corresponding to the given ``values``.
        values: `torch.Tensor`
            Contains the individual true scores corresponding to the given ``times``.
        x: `list` [`float`]
            The individual parameters.

        Returns
        -------
        err: `torch.Tensor`
            Model values minus real values.
        """
        xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
        tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

        if self.model_name == 'univariate':
            individual_parameters = {'xi': xi, 'tau': tau}
            err = model.compute_individual_tensorized(times, individual_parameters) - values
        else:
            sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
            individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
            err = model.compute_individual_tensorized(times, individual_parameters) - values
        return err

    def _get_regularity(self, model, x):
        """
        Compute the regularity of a patient given his individual parameters for a given model.

        Parameters
        ----------
        model: Leaspy model class object
            Model used to compute the group average parameters.
        x: `list` [`float`]
            The individual parameters.

        Returns
        -------
        regularity: `torch.Tensor`
            Regularity of the patient corresponding to the given individual parameters.
        """
        xi = torch.tensor(x[0], dtype=torch.float32)
        tau = torch.tensor(x[1], dtype=torch.float32)
        if self.model_name == 'univariate':
            iterates = zip(['xi', 'tau'], (xi, tau))
        else:
            sources = torch.tensor(x[2:], dtype=torch.float32)
            iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

        regularity = 0
        for key, value in iterates:
            mean = model.parameters["{0}_mean".format(key)]
            std = model.parameters["{0}_std".format(key)]
            regularity += torch.sum(model.compute_regularity_variable(value, mean, std))
        return regularity

    def _get_individual_parameters_patient(self, model, times, values):
        """
        Compute the individual parameter by minimizing the objective loss function with scipy solver.

        Parameters
        ----------
        model: Leaspy model class object
            Model used to compute the group average parameters.
        times: `torch.Tensor`
            Contains the individual ages corresponding to the given ``values``.
        values: `torch.Tensor`
            Contains the individual true scores corresponding to the given ``times``.

        Returns
        -------
            - tau - `float`
                Individual time-shift.
            - xi - `float`
                Individual log-acceleration.
            - sources - `list` [`float`]
                Individual space-shifts.
            - error - `torch.Tensor`
                Model values minus real values.
        """
        timepoints = times.reshape(1, -1)
        self._set_model_name(model.name)

        def obj(x, *args):
            """
            Objective loss function to minimize in order to get patient's individual parameters

            Parameters
            ----------
            x: `list` [`float`]
                Initialization of individual parameters
                By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)
            args:
                - model: leaspy model class object
                    Model used to compute the group average parameters.
                - timepoints: `torch.Tensor`
                    Contains the individual ages corresponding to the given ``values``
                - values: `torch.Tensor`
                    Contains the individual true scores corresponding to the given ``times``.

            Returns
            -------
            objective: `float`
                Value of the loss function.
            """

            # Parameters
            model, times, values = args

            # Attachment
            xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
            tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

            if self.model_name == 'univariate':
                individual_parameters = {'xi': xi, 'tau': tau}
                attachment = model.compute_individual_tensorized(times, individual_parameters)
                iterates = zip(['xi', 'tau'], (xi, tau))
            else:
                sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
                individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
                attachment = model.compute_individual_tensorized(times, individual_parameters)
                iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

            diff = attachment - values
            mask = (diff != diff)

            if self.loss == 'MSE':
                attachment = diff
                attachment[mask] = 0.  # Set nan to zero, not to count in the sum
                attachment = torch.sum(attachment ** 2) / (2. * model.parameters['noise_std'] ** 2)
            elif self.loss == 'crossentropy':
                attachment = torch.clamp(attachment, 1e-38, 1. - 1e-7)  # safety before taking the log
                neg_crossentropy = values * torch.log(attachment) + (1. - values) * torch.log(1. - attachment)
                neg_crossentropy[mask] = 0. # Set nan to zero, not to count in the sum
                attachment = -torch.sum(neg_crossentropy)
            # Regularity
            regularity = 0
            for key, value in iterates:
                mean = model.parameters["{0}_mean".format(key)]
                std = model.parameters["{0}_std".format(key)]
                regularity += torch.sum(model.compute_regularity_variable(value, mean, std))

            return (regularity + attachment).detach().tolist()

        def jacob(x, *args):
            """
            Jacobian of the objective loss function to minimize in order to get patient's individual parameters

            Parameters
            ----------
            x: `list` [`float`]
                Initialization of individual parameters
                By default x = [xi_mean, tau_mean] (+ [0.] * nber_of_sources if multivariate model)
            args:
                - model: leaspy model class object
                    Model used to compute the group average parameters.
                - timepoints: `torch.Tensor`
                    Contains the individual ages corresponding to the given ``values``
                - values: `torch.Tensor`
                    Contains the individual true scores corresponding to the given ``times``.

            Returns
            -------
            objective: `float`
                Value of the jacobian of the loss function.
            """
            # Parameters
            model, times, values = args

            # Attachment
            xi = torch.tensor([x[0]], dtype=torch.float32).unsqueeze(0)
            tau = torch.tensor([x[1]], dtype=torch.float32).unsqueeze(0)

            if self.model_name == 'univariate':
                individual_parameters = {'xi': xi, 'tau': tau}
                attachment = model.compute_individual_tensorized(times, individual_parameters)
                jacobian = model.compute_jacobian_tensorized(times, individual_parameters)
                iterates = zip(['xi', 'tau'], (xi, tau))
            else:
                sources = torch.tensor(x[2:], dtype=torch.float32).unsqueeze(0)
                individual_parameters = {'xi': xi, 'tau': tau, 'sources': sources}
                attachment = model.compute_individual_tensorized(times, individual_parameters)
                jacobian = model.compute_jacobian_tensorized(times, individual_parameters)
                iterates = zip(['xi', 'tau', 'sources'], (xi, tau, sources))

            attachment = attachment.unsqueeze(0)
            diff = attachment - values.unsqueeze(0)

            if self.loss == 'MSE':
                attachment = diff * jacobian
                mask = (attachment != attachment)
                attachment[mask] = 0.
                # Set nan to zero, not to count in the sum
                attachment = torch.sum(attachment, dim=(0, 2, 3)) / (model.parameters['noise_std'] ** 2)
            elif self.loss == 'crossentropy':
                attachment = torch.clamp(attachment, 1e-38, 1. - 1e-7)  # safety before dividing
                neg_crossentropy = diff / (attachment * (1. - attachment))
                neg_crossentropy = neg_crossentropy * jacobian
                mask = (neg_crossentropy != neg_crossentropy)
                neg_crossentropy[mask] = 0.  # Set nan to zero, not to count in the sum
                attachment = torch.sum(neg_crossentropy, dim=(0, 2, 3))
            # Regularity
            regularities = []
            for i, (key, value) in enumerate(iterates):
                mean = model.parameters["{0}_mean".format(key)]
                std = model.parameters["{0}_std".format(key)]
                regularities.append((value - mean) / (std ** 2))
            regularity = torch.cat(regularities, dim=1).sum(dim=0)

            return (regularity + attachment).detach().tolist()

        initial_value = self._initialize_parameters(model)
        if self.algo_parameters['use_jacobian']:
            res = minimize(obj, jac=jacob,
                           x0=initial_value,
                           args=(model, timepoints, values),
                           **self.minimize_kwargs
                           )
        else:
            res = minimize(obj,
                           x0=initial_value,
                           args=(model, timepoints, values),
                           **self.minimize_kwargs
                           )

        if res.success is not True and self.verbose:
            print(res.success, res)

        xi_f, tau_f, sources_f = res.x[0], res.x[1], res.x[2:]
        err_f = self._get_attachment(model, times.unsqueeze(0), values, res.x)

        return (tau_f, xi_f, sources_f), err_f  # TODO depends on the order

    def _get_individual_parameters(self, model, data):
        """
        Compute individual parameters of all patients given a leaspy model & a leaspy dataset.

        Parameters
        ----------
        model: leaspy model class object
            Model used to compute the group average parameters.
        data: leaspy.io.data.dataset.Dataset class object
            Contains the individual scores.

        Returns
        -------
        out: `dict` ['str`, `torch.Tensor`]
            Contains the individual parameters of all patients.
        """

        individual_parameters = IndividualParameters()

        p_names = model.get_individual_variable_name()

        for iter in range(data.n_individuals):
            times = data.get_times_patient(iter)  # torch.Tensor
            values = data.get_values_patient(iter)  # torch.Tensor
            idx = data.indices[iter]

            ind_patient, err = self._get_individual_parameters_patient(model, times, values)
            ind_p = {k: v for k, v in zip(p_names, ind_patient)}
            individual_parameters.add_individual_parameters(str(idx), ind_p)

        return individual_parameters

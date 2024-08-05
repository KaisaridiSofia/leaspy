from scipy.optimize import minimize
import torch

from .abstract_personalize_algo import AbstractPersonalizeAlgo
from ...io.outputs.individual_parameters import IndividualParameters


class ScipyMinimize(AbstractPersonalizeAlgo):

    def __init__(self, settings):
        super(ScipyMinimize, self).__init__(settings)

        self.verbose = False
        if "verbose" in self.algo_parameters:
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
            x += [0. for _ in range(model.source_dimension)]
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
        individual_params_f = self._pull_individual_parameters(x, model)
        err = model.compute_individual_tensorized(times, individual_params_f) - values
        return err

    def _get_regularity(self, model, individual_parameters):
        """
        Compute the regularity of a patient given his individual parameters for a given model.

        Parameters
        ----------
        model : :class:`.AbstractModel`
            Model used to compute the group average parameters.

        individual_parameters : dict[str, :class:`torch.Tensor` [n_ind,n_dims_param]]
            Individual parameters as a dict

        Returns
        -------
        regularity : :class:`torch.Tensor` [n_individuals]
            Regularity of the patient(s) corresponding to the given individual parameters.
            (Sum on all parameters)

        regularity_grads : dict[param_name: str, :class:`torch.Tensor` [n_individuals, n_dims_param]]
            Gradient of regularity term with respect to individual parameters.
        """

        regularity = 0
        regularity_grads = {}

        for param_name, param_val in individual_parameters.items():
            # priors on this parameter
            priors = dict(
                mean = model.parameters[param_name+"_mean"],
                std = model.parameters[param_name+"_std"]
            )

            # summation term
            regularity += model.compute_regularity_variable(param_val, **priors).sum(dim=1)

            # derivatives: <!> formula below is for Normal parameters priors only
            # TODO? create a more generic method in model `compute_regularity_variable_gradient`? but to do so we should probably wait to have some more generic `compute_regularity_variable` as well (at least pass the parameter name to this method to compute regularity term)
            regularity_grads[param_name] = (param_val - priors['mean']) / (priors['std']**2)

        return (regularity, regularity_grads)


    def _pull_individual_parameters(self, x, model):
        """
        Get individual parameters as a dict[param_name: str, :class:`torch.Tensor` [1,n_dims_param]]
        from a condensed array-like version of it

        (based on the conventional order defined in :meth:`._initialize_parameters`)
        """
        tensorized_params = torch.tensor(x, dtype=torch.float32).view((1,-1)) # 1 individual

        # <!> order + rescaling of parameters
        individual_parameters = {
            'xi': tensorized_params[:,[0]] * model.parameters['xi_std'],
            'tau': tensorized_params[:,[1]] * model.parameters['tau_std'],
        }
        if 'univariate' not in model.name and model.source_dimension > 0:
            individual_parameters['sources'] = tensorized_params[:, 2:] * model.parameters['sources_std']

        return individual_parameters

    def _get_normalized_grad_tensor_from_grad_dict(self, dict_grad_tensors, model):
        """
        From a dict of gradient tensors per param (without normalization),
        returns the full tensor of gradients (= for all params, consecutively):
            * concatenated with conventional order of x0
            * normalized because we derive w.r.t. "standardized" parameter (adimensional gradient)
        """
        to_cat = [
            dict_grad_tensors['xi'] * model.parameters['xi_std'],
            dict_grad_tensors['tau'] * model.parameters['tau_std']
        ]
        if 'univariate' not in model.name and model.source_dimension > 0:
            to_cat.append( dict_grad_tensors['sources'] * model.parameters['sources_std'] )

        return torch.cat(to_cat, dim=-1).transpose(0, -1).squeeze(-1) # 1 individual at a time


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
            individual_parameters = self._pull_individual_parameters(x, model)

            attachment = model.compute_individual_tensorized(times, individual_parameters)

            if self.loss == 'MSE':
                diff = attachment - values
                mask = (diff != diff)
                attachment = diff
                attachment[mask] = 0.  # Set nan to zero, not to count in the sum
                attachment = torch.sum(attachment ** 2) / (2. * model.parameters['noise_std'] ** 2)
            elif self.loss == 'crossentropy':
                attachment = torch.clamp(attachment, 1e-38, 1. - 1e-7)  # safety before taking the log
                neg_crossentropy = values * torch.log(attachment) + (1. - values) * torch.log(1. - attachment)
                neg_crossentropy[neg_crossentropy != neg_crossentropy] = 0. # Set nan to zero, not to count in the sum
                attachment = -torch.sum(neg_crossentropy)
            elif self.loss == 'ordinal':
                max_level = max([feat["nb_levels"] for feat in model.ordinal_infos])

                s = list(attachment.shape)
                s[-1] = 1
                ones = torch.ones(size=tuple(s)).float()
                pred = torch.cat([ones, attachment], dim=-1)
                vals = values.long().clamp(0, max_level)
                t = torch.eye(max_level + 1)
                for k in range(max_level):
                    t[k, k + 1] = -1.
                vals[vals != vals] = 0
                vals = t[vals]

                mask = (values == values)

                LL = -(torch.log((pred * vals).sum(dim=-1)).clamp(-20., 0.))
                attachment = torch.sum(mask.float() * LL,)

            # Regularity
            regularity, _ = self._get_regularity(model, individual_parameters)

            return (regularity + attachment).item()

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
            individual_parameters = self._pull_individual_parameters(x, model)
            attachment = model.compute_individual_tensorized(times, individual_parameters)
            grads = model.compute_jacobian_tensorized(times, individual_parameters)
            # put derivatives consecutively in the right order and drop ind level
            # --> output shape [n_tpts, n_fts [, n_ordinal_lvls], n_dims_params]
            jacobian = self._get_normalized_grad_tensor_from_grad_dict(grads, model)

            attachment = attachment

            if self.loss == 'MSE':
                diff = attachment - values
                attachment = diff * jacobian
                mask = (attachment != attachment)
                attachment[mask] = 0.
                # Set nan to zero, not to count in the sum
                attachment = torch.sum(attachment, dim=(1, 2)) / (model.parameters['noise_std'] ** 2)
            elif self.loss == 'crossentropy':
                diff = attachment - values
                attachment = torch.clamp(attachment, 1e-38, 1. - 1e-7)  # safety before dividing
                neg_crossentropy = diff / (attachment * (1. - attachment))
                neg_crossentropy = neg_crossentropy * jacobian
                mask = (neg_crossentropy != neg_crossentropy)
                neg_crossentropy[mask] = 0.  # Set nan to zero, not to count in the sum
                attachment = torch.sum(neg_crossentropy, dim=(1, 2))
            elif self.loss == 'ordinal':
                max_level = max([feat["nb_levels"] for feat in model.ordinal_infos])

                s = list(attachment.shape)
                s[-1] = 1
                ones = torch.ones(size=tuple(s)).float()

                s = list(jacobian.shape)
                s[-1] = 1
                zeros = torch.zeros(size=tuple(s)).float()

                pred = torch.cat([ones, attachment], dim=-1)
                jac = torch.cat([zeros, jacobian], dim=-1)
                vals = values.long().clamp(0, max_level)
                t = torch.eye(max_level + 1)
                for k in range(max_level):
                    t[k, k + 1] = -1.
                vals[vals != vals] = 0
                vals = t[vals]

                mask = (values == values)

                LL = - (jac * vals).sum(dim=-1) / (pred * vals).sum(dim=-1)
                attachment = torch.sum(mask.float() * LL, dim=(1, 2))

            # Regularity
            _, regularity_grads = self._get_regularity(model, individual_parameters)
            regularity = self._get_normalized_grad_tensor_from_grad_dict(regularity_grads, model)


            return (regularity + attachment).detach().numpy()

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

        individual_params_f = self._pull_individual_parameters(res.x, model)
        individual_params_f = {
            k: v.item() if k != 'sources' else v.detach().squeeze(0).tolist()
            for k, v in individual_params_f.items()
        }

        if self.loss == "MSE":
            err_f = self._get_attachment(model, times.unsqueeze(0), values, res.x)
        else:
            err_f = "TODO compute_attachment for other losses"

        return individual_params_f, err_f  # TODO depends on the order

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

        for iter in range(data.n_individuals):
            times = data.get_times_patient(iter)  # torch.Tensor
            values = data.get_values_patient(iter)  # torch.Tensor
            idx = data.indices[iter]

            ind_patient, err = self._get_individual_parameters_patient(model, times, values)
            individual_parameters.add_individual_parameters(str(idx), ind_patient)

        return individual_parameters

from . import UnivariateModel, MultivariateModel, MultivariateParallelModel, MultivariateIPMixtureModel


class ModelFactory:
    """
    This class perfomrs the transition from the model's name to model object

    Methods
    -------
    model(name)
        Return the model object corresponding to 'name' arg
    """

    @staticmethod
    def model(name, **kwargs):
        """
        Return the model object corresponding to 'name' arg - check name type and value

        Parameters
        ----------
        name: str
            The model's name

        Returns
        -------
        a child class object of leaspy.model.AbstractModel class object determined by 'name'
        """
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name == 'univariate':
            return UnivariateModel(name)
        elif name == 'logistic' or name == 'linear' or name == 'mixed_linear-logistic':
            return MultivariateModel(name)
        elif name == 'logistic_mixture':
            return MultivariateIPMixtureModel(name, **kwargs)
        elif name == 'logistic_parallel':
            return MultivariateParallelModel(name)
        else:
            raise ValueError("The name of the model you are trying to create does not exist! " +
                             "It should be `logistic` or `logistic_parallel`")

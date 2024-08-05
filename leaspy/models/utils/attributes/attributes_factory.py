from . import AttributesLogisticParallel, AttributesLogistic, AttributesLinear, AttributesUnivariate, AttributesLogisticOrdinal


class AttributesFactory:

    @staticmethod
    def attributes(name, dimension, source_dimension, ordinal_infos=None):
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")
        if not ordinal_infos is None:
            return AttributesLogisticOrdinal(dimension, source_dimension, ordinal_infos)
        elif name == 'univariate':
            return AttributesUnivariate()
        elif name == 'logistic':
            return AttributesLogistic(dimension, source_dimension)
        elif name == 'logistic_mixture':
            return AttributesLogistic(dimension, source_dimension)
        elif name == 'logistic_parallel':
            return AttributesLogisticParallel(dimension, source_dimension)
        elif name == 'linear':
            return AttributesLinear(dimension, source_dimension)
        elif name == 'mixed_linear-logistic':
            return AttributesLogistic(dimension, source_dimension)  # TODO mixed check
        else:
            raise ValueError(
                "The name {} you provided for the attributes is not related to an attribute class".format(name))

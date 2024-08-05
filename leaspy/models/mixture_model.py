from leaspy.models.base import BaseModel
from leaspy.models.factory import ModelFactory

class MixtureModel(BaseModel):

    """
    Contains the common attributes & methods of the different probabilistic models.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    base_model: str
        The base model for the mixture.
    n_clusters: int
        The number of models in the mixture.
    **kwargs
        Hyperparameters for the base model.

    Attributes
    ----------
    """

    def __init__(self, name:str, base_model:str, n_clusters:int, **kwargs):
        super().__init__(name)
        self.n_clusters = n_clusters
        self.models = [ModelFactory(base_model, **kwargs)]
        self.state

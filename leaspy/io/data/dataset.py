import pandas as pd
import torch


class Dataset:
    def __init__(self, data, model=None, algo=None):
        # TODO : Change in Pytorch
        self.timepoints = None
        self.values = None
        self.mask = None
        self.headers = data.headers
        self.n_individuals = None
        self.nb_observations_per_individuals = None
        self.max_observations = None
        self.dimension = None
        self.n_visits = None
        self.individual_parameters = None
        self.indices = list(data.individuals.keys())
        self.L2_norm = None

        if model is not None:
            self._check_model_compatibility(data, model)
        if algo is not None:
            self._check_algo_compatibility(data, algo)

        self._construct_values(data)
        self._construct_timepoints(data)
        self._compute_L2_norm()

    def _construct_values(self, data):

        batch_size = data.n_individuals
        x_len = [len(_.timepoints) for _ in data]
        channels = data.dimension
        values = torch.zeros((batch_size, max(x_len), channels), dtype=torch.float32)
        # padding_mask = np.zeros((batch_size, max(x_len), channels))
        padding_mask = torch.zeros((batch_size, max(x_len), channels), dtype=torch.float32)

        # TODO missing values in mask ?

        for i, d in enumerate(x_len):
            indiv_values = torch.tensor(data[i].observations, dtype=torch.float32)
            values[i, 0:d, :] = indiv_values
            padding_mask[i, 0:d, :] = 1

        mask_missingvalues = 1 - torch.isnan(values) * 1
        mask = padding_mask * mask_missingvalues

        # values[np.array(1 - mask_missingvalues, dtype=bool)] = 0 # Set values of missing values to 0.0
        values[torch.isnan(values)] = 0  # Set values of missing values to 0.0

        self.n_individuals = batch_size
        self.max_observations = max(x_len)
        self.nb_observations_per_individuals = x_len
        self.dimension = channels
        self.values = values
        self.mask = mask
        self.n_visits = data.n_visits
        # self.n_observations = int(np.sum(mask))
        self.n_observations_per_ind_per_ft = mask.sum(dim=1)  # 2D int tensor of shape(n_individuals,dimension)
        self.n_observations_per_ft = self.n_observations_per_ind_per_ft.sum(dim=0)  # 1D int tensor of shape(dimension,)
        self.n_observations = int(mask.sum().item())

    def _construct_timepoints(self, data):
        self.timepoints = torch.zeros([self.n_individuals, self.max_observations], dtype=torch.float32)
        x_len = [len(_.timepoints) for _ in data]
        for i, d in enumerate(x_len):
            self.timepoints[i, 0:d] = torch.tensor(data[i].timepoints, dtype=torch.float32)

    def _compute_L2_norm(self):
        self.L2_norm = torch.sum(torch.sum(self.values * self.values * self.mask.float(), dim=2))

    def get_times_patient(self, i):
        return self.timepoints[i, :self.nb_observations_per_individuals[i]]

    def get_values_patient(self, i):
        values = self.values[i, :self.nb_observations_per_individuals[i], :]
        # mask = self.mask[i].clone().cpu().detach().numpy()[:values.shape[0],:]
        mask = self.mask[i].clone().cpu().detach()[:values.shape[0], :]
        # mask[mask==0] = np.nan
        mask[mask == 0] = float('NaN')
        values_with_na = values * mask
        return values_with_na

    @staticmethod
    def _check_model_compatibility(data, model):
        if model.dimension is None:
            return
        if data.dimension != model.dimension:
            raise ValueError(f"Unmatched dimensions. Model {model.dimension} ≠ {data.dimension} Data ")

    @staticmethod
    def _check_algo_compatibility(data, algo):
        return

    def to_pandas(self):
        # TODO : @Raphael : On est oblige de garder une dependance pandas ? Je crois que c'est utilise juste pour l'initialisation
        # TODO : Si fait comme ca, il faut preallouer la memoire du dataframe a l'avance!
        # du modele multivarie. On peut peut-etre s'en passer?
        df = pd.DataFrame()

        for i, idx in enumerate(self.indices):
            times = self.get_times_patient(i).cpu().numpy()
            x = self.get_values_patient(i).cpu().numpy()
            df_patient = pd.DataFrame(data=x, index=times.reshape(-1), columns=self.headers)
            df_patient.index.name = 'TIME'
            df_patient['ID'] = idx
            df = pd.concat([df, df_patient])

        # df.columns = ['ID', 'TIME'] + self.headers
        df.reset_index(inplace=True)

        return df

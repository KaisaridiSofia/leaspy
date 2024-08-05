import csv
import os
import time

from leaspy.io.logs.visualization.plotter import Plotter


class FitOutputManager():
    """
    FitOutputManager class object
    Class used by AbstractAlgo (and its child classes) to display plots and statistics and to save

    Attributes
    ----------
    path_output: str
        Path of the folder containing all the outputs
    path_plot: str
        Path of the subfolder of path_output containing the logs plots
    path_plot_convergence_model_parameters_1: str
        Path of the first plot of the convergence of the model's parameters (in the subfolder path_plot)
    path_plot_convergence_model_parameters_2: str
        Path of the second plot of the convergence of the model's parameters (in the subfolder path_plot)
    path_plot_patients: str
        Path of the subfolder of path_plot containing the plot of the reconstruction of the patients' longitudinal
        trajectory by the model
    path_save_model_parameters_convergence: str
        Path of the subfolder of path_output containing the progression of the model's parameters convergence
    periodicity_plot: int (default 100)
        Set the frequency of the display of the plots
    periodicity_print: int
        Set the frequency of the display of the statistics
    periodicity_save: int
        Set the frequency of the saves of the model's parameters & the realizations
    plot_options: dict
        Contain all the additional information (for now contain only the number of displayed patients by the method
        plot_patient_reconstructions - which is 5 by default)
    plotter: a leaspy.utils.logs.visualisation.plotter.Plotter class object
        class object used to call visualization methods
    time: scalar
        Used to display the duration between two visualization prints

    Methods
    -------
    iteration(algo, data, model, realizations)
        Call methods to save state of the running computation, display statistics & plots if the current iteration
        is a multiple of periodicity_print, periodicity_plot or periodicity_save
    plot_convergence_model_parameters(model)
        Plot the convergence of the model parameters (calling the Plotter class object)
    plot_patient_reconstructions(iteration, data, model, realizations)
        Plot on the same graph for several patients their real longitudinal values and their reconstructions by the model
    print_algo_statistics(algo)
        Print algorithm's statistics
    print_model_statistics(model)
        Print model's statistics
    print_time()
        Display duration since last print
    save_model_parameters_convergence(iteration, model)
        Save the current state of the model's parameters.
        The path is given by the attribute path_save_model_parameters_convergence
    save_realizations(iteration, realizations)
        Save the current realizations.
        The path is given by the attribute path_save_model_parameters_convergence
    """

    # TODO: add a loading bar for a run

    def __init__(self, outputs):
        """
        Process initializer function that is called by class FitOutputManager

        Parameters
        ----------
        outputs: a leaspy.io.settings.outputs_settings.OutputsSettings class object
            Initialize the FitOuputManager class attributes, like the logs paths, the console print periodicity and so forth.
        """
        self.path_output = outputs.root_path
        self.path_plot = outputs.plot_path
        self.path_plot_patients = outputs.patients_plot_path
        self.path_plot_convergence_model_parameters_1 = os.path.join(outputs.plot_path, "convergence_1.pdf")
        self.path_plot_convergence_model_parameters_2 = os.path.join(outputs.plot_path, "convergence_2.pdf")
        self.path_save_model_parameters_convergence = outputs.parameter_convergence_path
        self.periodicity_plot = outputs.plot_periodicity
        self.periodicity_print = outputs.console_print_periodicity
        self.periodicity_save = outputs.save_periodicity

        # Options
        # TODO : Maybe add to the outputs reader
        self.plot_options = {}
        self.plot_options['maximum_patient_number'] = 5
        self.plotter = Plotter()

        self.time = time.time()

    def iteration(self, algo, data, model, realizations, print=True):
        """
        Call methods to save state of the running computation, display statistics & plots if the current iteration
        is a multiple of periodicity_print, periodicity_plot or periodicity_save

        Parameters
        ----------
        algo: a leaspy.algo.abstract_algo.AbstractAlgo class object (or one of its child class)
            The running algorithm
        data: a leaspy.io.data.data.Data leaspy class object
            The data used by the computation
        model: a leaspy.model.abstract_model.AbstractModel class object (or one of its child class)
            The model used by the computation
        realizations: a leaspy.utils.realizations.collection_realization.CollectionRealization class object
            Current state of the realizations
        """
        iteration = algo.current_iteration

        if self.periodicity_print is not None and print:
            if iteration % self.periodicity_print == 0:
                self.print_algo_statistics(algo)
                self.print_model_statistics(model)
                self.print_time()

        if self.path_output is None:
            return

        if self.periodicity_save is not None:
            if iteration % self.periodicity_save == 0:
                self.save_model_parameters_convergence(iteration, model)
                # self.save_model(model)

        if self.periodicity_plot is not None:
            if iteration % self.periodicity_plot == 0:
                self.plot_patient_reconstructions(iteration, data, model, realizations)
                self.plot_convergence_model_parameters(model)

        if (algo.algo_parameters['n_iter'] - iteration) < 100:
            self.save_realizations(iteration, realizations)

    ########
    ## Printing methods
    ########

    def print_time(self):
        """
        Display the duration since the last print
        """
        current_time = time.time()
        print("Duration since last print : {0}s".format(round(current_time - self.time), 4))
        self.time = current_time

    def print_model_statistics(self, model):
        """
        Print model's statistics

        Parameters
        ----------
        model: a leaspy.model.abstract_model.AbstractModel class object (or one of its child class)
            The model used by the computation
        """
        print(model)

    def print_algo_statistics(self, algo):
        """
        Print algorithm's statistics

        Parameters
        ----------
        algo: a leaspy.algo.abstract_algo.AbstractAlgo class object (or one of its child class)
            The running algorithm
        """
        print(algo)

    ########
    ## Saving methods
    ########

    def save_model_parameters_convergence(self, iteration, model):
        """
        Save the current state of the model's parameters

        Parameters
        ----------
        iteration: int
            The current iteration
        model: a leaspy.model.abstract_model.AbstractModel class object (or one of its child class)
            The model used by the computation
        """
        model_parameters = model.parameters

        # TODO maybe better way ???
        model_parameters_save = model_parameters.copy()

        # TODO I Stopped here, 2d array saves should be fixed.

        # Transform the types
        for key, value in model_parameters.items():

            if value.ndim > 1:
                if key in ["betas", "deltas"]:
                    model_parameters_save.pop(key)
                    for column in range(value.shape[1]):
                        model_parameters_save["{0}_{1}".format(key, column)] = value[:, column].tolist()
                # P0, V0
                elif value.shape[0] == 1 and len(value.shape) > 1:
                    model_parameters_save[key] = value[0].tolist()
            elif value.ndim == 1:
                model_parameters_save[key] = value.tolist()
            else:  # ndim == 0
                model_parameters_save[key] = [value.tolist()]

        # Save the dictionnary
        for key, value in model_parameters_save.items():
            path = os.path.join(self.path_save_model_parameters_convergence, key + ".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                writer.writerow([iteration] + value)

    def save_realizations(self, iteration, realizations):
        """
        Save the current realizations. The path is given by the attribute path_save_model_parameters_convergence

        Parameters
        ----------
        iteration: int
            The current iteration
        realizations: a leaspy.utils.realizations.collection_realization.CollectionRealization class object
            Current state of the realizations
        """
        for name in ['xi', 'tau']:
            value = realizations[name].tensor_realizations.squeeze(1).detach().tolist()
            path = os.path.join(self.path_save_model_parameters_convergence, name + ".csv")
            with open(path, 'a', newline='') as filename:
                writer = csv.writer(filename)
                # writer.writerow([iteration]+list(model_parameters.values()))
                writer.writerow([iteration] + value)
        if "sources" in realizations.reals_ind_variable_names:
            for i in range(realizations['sources'].tensor_realizations.shape[1]):
                value = realizations['sources'].tensor_realizations[:, i].detach().tolist()
                path = os.path.join(self.path_save_model_parameters_convergence, 'sources' + str(i) + ".csv")
                with open(path, 'a', newline='') as filename:
                    writer = csv.writer(filename)
                    # writer.writerow([iteration]+list(model_parameters.values()))
                    writer.writerow([iteration] + value)

    ########
    ## Plotting methods
    ########

    def plot_convergence_model_parameters(self, model):
        """
        Plot the convergence of the model parameters (calling the Plotter class object)

        Parameters
        ----------
        model: a leaspy.model.abstract_model.AbstractModel class object (or one of its child class)
            The model used by the computation
        """
        self.plotter.plot_convergence_model_parameters(self.path_save_model_parameters_convergence,
                                                       self.path_plot_convergence_model_parameters_1,
                                                       self.path_plot_convergence_model_parameters_2,
                                                       model)

    def plot_model_average_trajectory(self, model):
        raise NotImplementedError

    def plot_patient_reconstructions(self, iteration, data, model, realizations):
        """
        Plot on the same graph for several patients their real longitudinal values and their reconstructions by the model

        Parameters
        ----------
        iteration: int
            The current iteration
        data: a leaspy.io.data.data.Data leaspy class object
            The data used by the computation
        model: a leaspy.model.abstract_model.AbstractModel class object (or one of its child class)
            The model used by the computation
        realizations: a leaspy.utils.realizations.collection_realization.CollectionRealization class object
            Current state of the realizations
        """
        path_iteration = os.path.join(self.path_plot_patients, 'plot_patients_{0}.pdf'.format(iteration))
        param_ind = model.get_param_from_real(realizations)
        self.plotter.plot_patient_reconstructions(path_iteration, data, model, param_ind,
                                                  self.plot_options['maximum_patient_number'])

        """
        colors = cm.rainbow(np.linspace(0, 1, self.plot_options['maximum_patient_number']+2))
        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1, 1)

        for i, idx in enumerate(data.indices):
            model_value = model.compute_individual(data[idx], reals_pop, reals_ind[idx])
            score = data[idx].tensor_observations
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(data[idx].tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--',
                    marker='o')

            if i > self.plot_options['maximum_patient_number']:
                break

        # Plot average model
        tensor_timepoints = torch.Tensor(np.linspace(data.time_min, data.time_max, 40).reshape(-1,1))
        model_average = model.compute_average(tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth=4, alpha=0.3)

        plt.savefig(os.path.join(self.path_plot_patients,'plot_patients_{0}.pdf'.format(iteration)))
        plt.close()
        """

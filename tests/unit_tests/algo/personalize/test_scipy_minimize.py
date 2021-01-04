import os
import unittest
import torch
import numpy as np

from tests import test_data_dir
from leaspy.api import Leaspy
from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.io.settings.algorithm_settings import AlgorithmSettings


class ScipyMinimizeTest(unittest.TestCase):

    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'use_jacobian': False})
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

    def test_get_model_name(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('name')

        self.assertEqual(algo.model_name, 'name')

    def test_initialize_parameters(self, ):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        univariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'univariate.json')
        univariate_model = Leaspy.load(univariate_path)
        param = algo._initialize_parameters(univariate_model.model)

        self.assertEqual(param, [torch.tensor([-1.0]), torch.tensor([70.0])])

        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        multivariate_model = Leaspy.load(multivariate_path)
        param = algo._initialize_parameters(multivariate_model.model)
        self.assertEqual(param, [torch.tensor([0.0]), torch.tensor([75.2]), torch.tensor([0.]), torch.tensor([0.])])

    def test_get_attachment(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('logistic')

        times = torch.tensor([70, 80])
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        individual_parameters = [0.0, 75.2, 0., 0.]

        err = algo._get_attachment(leaspy.model, times, values, individual_parameters)

        output = torch.tensor([[
            [-0.4705, -0.3278, -0.3103, -0.4477],
            [0.6059,  0.0709,  0.3537,  0.4523]]])
        self.assertEqual(torch.is_tensor(err), True)
        self.assertAlmostEqual(torch.sum((err - output)**2).item(), 0, delta=10e-8)

    def test_get_regularity(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('logistic')

        individual_parameters = [0.0, 75.2, 0., 0.]

        reg = algo._get_regularity(leaspy.model, individual_parameters)
        self.assertEqual(torch.is_tensor(reg), True)
        output = torch.tensor([4.0264])
        self.assertAlmostEqual(torch.sum((reg - output)**2).item(), 0, delta=10e-9)

    def test_get_individual_parameters_patient_univariate(self):
        # TODO
        return 0

    def test_get_individual_parameters_patient_multivariate(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0)
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.assertAlmostEqual(individual_parameters[0], 78.93283994514304, delta=tol)
        self.assertAlmostEqual(individual_parameters[1], -0.07679465847751077, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[0], -0.07733279, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[1], -0.57428166, delta=tol)

        err_expected = torch.tensor([[
            [-0.4958, -0.3619, -0.3537, -0.4497],
            [0.1650, -0.0948,  0.1361, -0.1050]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0.5, 0.4, 0.4, float('nan')], [0.3, float('nan'), float('nan'), 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.assertAlmostEqual(individual_parameters[0], 78.82484683798302, delta=tol)
        self.assertAlmostEqual(individual_parameters[1], -0.07808162619234782, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[0], -0.17007795, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[1], -0.63483322, delta=tol)

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        self.assertEqual(torch.all(torch.eq(torch.isnan(err), nan_positions)), True)

        err[err != err] = 0.
        err_expected = torch.tensor([[
            [-0.4957, -0.3613, -0.3516, 0.],
            [0.1718, 0., 0., -0.0796]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

    def test_get_individual_parameters_patient_crossentropy(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0, loss="crossentropy")
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0., 1., 0., 1.], [0., 1., 1., 1.]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.assertAlmostEqual(individual_parameters[0], 69.91258208274421, delta=tol)
        self.assertAlmostEqual(individual_parameters[1], -0.1446537485712681, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[0], -0.16517799, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[1], -0.82381726, delta=tol)

        err_expected = torch.tensor([[
            [ 0.3184, -0.8266,  0.2943, -0.8065],
            [ 0.9855, -0.4526, -0.2114, -0.0048]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0., 1., 0., float('nan')], [0., float('nan'), float('nan'), 1.]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.assertAlmostEqual(individual_parameters[0], 76.57318992643758, delta=tol)
        self.assertAlmostEqual(individual_parameters[1], -0.06489393539830259, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[0], -0.08735905, delta=tol)
        self.assertAlmostEqual(individual_parameters[2].tolist()[1], -0.37562645, delta=tol)

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        self.assertEqual(torch.all(torch.eq(torch.isnan(err), nan_positions)), True)

        err[err != err] = 0.
        err_expected = torch.tensor([[
            [ 0.0150, -0.9417,  0.0752,     0.],
            [ 0.7702,    0.,     0., -0.3218]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

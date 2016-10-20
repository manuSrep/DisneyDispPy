import unittest
import sys
import os

import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))
from DisneyDisp import edge_confidence, sample_radiance, score_computation, estimate_disp



class TestDisparityEstimation(unittest.TestCase):

    def setUp(self):
        # We create an artificial lf
        imgs = np.zeros((50,100,150), dtype=np.uint8)
        for i in range(50):
            for j in range(2):
                rr, cc = line(0, 10+j+i, 99, 10+j+i)
                imgs[i, rr, cc] = 75
            for j in range(5):
                rr, cc = line(0, 12+j+i, 99, 12+j+i)
                imgs[i, rr, cc] = 125
            for j in range(2):
                rr, cc = line(0, 17+j+i, 99, 17+j+i)
                imgs[i, rr, cc] = 75
            for j in range(5):
                rr, cc = line(0, 20+j+2*i, 99, 20+j+2*i)
                imgs[i, rr, cc] = 175
            for j in range(10):
                rr, cc = line(0, 35+j+2*i, 99, 35+j+2*i)
                imgs[i, rr, cc] = 250
            #ski.io.imsave("img_{i}.png".format(i=i), imgs[i])

        # We create epis out of it
        self.epis = np.zeros((100,50,150), dtype=np.uint8)
        for i in range(100):
            self.epis[i] = np.reshape(imgs[:,i], (50,150))
            #ski.io.imsave("epi_{i}.png".format(i=i), epis[i])
        self.epi = self.epis[50]


    def test_score_computation_noisefree(self):
        epi = self.epi
        s_hat = 25
        min_disp = -5
        max_disp = 0
        stepsize = 1
        DB = np.full((150,2), fill_value=-np.inf)
        DB[:,1] = np.inf
        M = np.full((150), fill_value=True, dtype=np.bool)
        D = np.full((150), fill_value=np.nan, dtype=np.float32)

        Ce, Me = edge_confidence(self.epi, window=9, threshold=0.02)
        R, Mr, disp_range, plots = sample_radiance(epi, s_hat, min_disp, max_disp, stepsize, DB, Me[s_hat], DEBUG=True)

        S_norm, R_bar = score_computation(R, epi, s_hat, Me[s_hat], Mr, h=0.02, NOISEFREE=True)

        D, R_bar_best, S_argmax = estimate_disp(D, S_norm, Me[s_hat], R_bar, disp_range)

        self.assertTrue(np.all(np.isnan(D[~Me[s_hat]])))


    def doCleanups(self):
        pass


if __name__ == '__main__':
    unittest.main()
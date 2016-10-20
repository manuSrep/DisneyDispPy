import unittest
import sys
import os

import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))
from DisneyDisp import edge_confidence, sample_radiance, score_computation, estimate_disp, propagation



class TestPopagation(unittest.TestCase):

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

        self.disp = np.zeros((100,50,150), dtype=np.float32)
        self.disp[self.epis == 75] = -1
        self.disp[self.epis == 125] = -1
        self.disp[self.epis == 175] = -2
        self.disp[self.epis == 250] = -2
        self.disp[self.disp == 0] = np.nan


    def test_propagation(self):
        disp = np.full((100,50,150), fill_value=np.nan, dtype=np.float32)
        disp[:,25] = self.disp[:,25]
        disp[disp == 0] = np.nan
        Ds, plot = propagation(disp, self.epis, self.epis[:,25], 25, threshold=0.1, DEBUG=True)
        #plt.imsave("propagation.png", Ds[50], )
        np.testing.assert_array_equal(Ds[50],self.disp[50])



    def doCleanups(self):
        pass


if __name__ == '__main__':
    unittest.main()
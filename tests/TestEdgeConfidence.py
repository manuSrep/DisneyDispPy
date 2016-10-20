import unittest
import sys
import os

import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))
from DisneyDisp import edge_confidence



class TestEdgeConfidence(unittest.TestCase):

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
        self.epi = self.epis[25]


    def test_EdgeConfidence(self):
        Ce, Me = edge_confidence(self.epi, window=9, threshold=0.02)
        #np.save("results/Ce.npy", Ce, allow_pickle=False)
        #np.save("results/Me.npy",Me, allow_pickle=False)
        #plt.imsave("Ce.png", Ce)
        #plt.imsave("Me.png", Me, cmap="gray")
        Ce_exp = np.load("results/Ce.npy", allow_pickle=False)
        Me_exp = np.load("results/Me.npy", allow_pickle=False)
        np.testing.assert_array_equal(Ce,Ce_exp)
        np.testing.assert_array_equal(Me, Me_exp)


    def doCleanups(self):
        pass

if __name__ == '__main__':
    unittest.main()
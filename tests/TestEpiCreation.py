import unittest
import sys
import os

import numpy as np
import h5py as h5
import skimage as ski
from skimage import draw, io


sys.path.append(os.path.abspath("../"))

from DisneyDisp import calculate_resolutions, downsample_lightfield, create_epis

class TestEpiCreation(unittest.TestCase):

    def setUp(self):
        pass

    def test_creation(self):

        # We create an artificial lf
        imgs = np.zeros((50,100,150), dtype=np.uint8)
        for i in range(50):
            rr1, cc1 = ski.draw.line(0, 25+2*i, 99, 25+2*i)
            rr2, cc2 = ski.draw.line(0, 75+i, 99, 75+i)
            imgs[i, rr1, cc1] = 255
            imgs[i, rr2, cc2] = 255
        lf = h5.File("test_lf.hdf5")
        grp = lf.create_group("lightfield")
        grp.create_dataset("100x150", data=imgs)
        grp.attrs.create("resolutions", [[100,150]])
        lf.close()

        create_epis("test_lf.hdf5", "test_epi.hdf5", "lightfield", hdf5_dataset_out="epis",  dtype=np.uint8, RGB=True)
        epi = h5.File("test_epi.hdf5")["epis/100x150"][...]
        #np.save("results/expected_epis.npy", epi, allow_pickle=False)
        epi_exp = np.load("results/expected_epis.npy", allow_pickle=False)
        np.testing.assert_array_equal(epi, epi_exp)


    def doCleanups(self):
        if os.path.isfile(("test_lf.hdf5")):
            os.remove("test_lf.hdf5")
        if os.path.isfile(("test_epi.hdf5")):
            os.remove("test_epi.hdf5")


if __name__ == '__main__':
    unittest.main()
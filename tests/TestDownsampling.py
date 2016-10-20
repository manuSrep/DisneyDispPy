import unittest
import sys
import os

import numpy as np
import h5py as h5
import skimage as ski
from skimage import draw, io


sys.path.append(os.path.abspath("../"))

from DisneyDisp import calculate_resolutions, downsample_lightfield

class TestDownsampling(unittest.TestCase):

    def setUp(self):
        pass


    def test_resolutions(self):
        # We need am artificial image we can downsample
        img = np.zeros((30, 40), dtype=np.uint8)
        rr1, cc1 = ski.draw.line(0, 10, 29, 10)
        rr2, cc2 = ski.draw.line(0, 15, 29, 15)
        img[rr1, cc1] = 255
        img[rr2, cc2] = 255
        res_exp = np.array([[30,40], [15,20]])
        res_new = calculate_resolutions(*img.shape, red_fac=2, min_res=11)
        np.testing.assert_array_equal(res_exp, res_new)

    def test_downsampling(self):

        # We create an artificial lf
        imgs = np.zeros((50,100,150), dtype=np.uint8)
        for i in range(50):
            rr1, cc1 = ski.draw.line(0, 25+2*i, 99, 25+2*i)
            rr2, cc2 = ski.draw.line(0, 75+i, 99, 75+i)
            imgs[i, rr1, cc1] = 255
            imgs[i, rr2, cc2] = 255
        lf = h5.File("test_lf.hdf5")
        lf.create_dataset("lightfield", data=imgs)
        lf.close()

        r_all = np.array([[100,150],[50,75],[25,38],[13,19]])
        downsample_lightfield("test_lf.hdf5", "down_lf.hdf5", "lightfield", r_all)

        lf_down = h5.File("down_lf.hdf5")["lightfield"]["25x38"]
        #np.save("results/expected_downsampled_lf.npy", lf_down, allow_pickle=False)
        lf_exp = np.load("results/expected_downsampled_lf.npy", allow_pickle=False)
        np.testing.assert_array_equal(lf_down, lf_exp)


    def doCleanups(self):
        if os.path.isfile(("test_lf.hdf5")):
            os.remove("test_lf.hdf5")
        if os.path.isfile(("down_lf.hdf5")):
            os.remove("down_lf.hdf5")


if __name__ == '__main__':
    unittest.main()
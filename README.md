# README

DisneyDisp is a free python implementation of *Scene Reconstruction from High Spatio-Angular Resolution Light Fields* originally developed by *Disney research*. [[1][]]. It enables disparity estimation of 3D lightfields by preserving fine details. For more details see also: <https://www.disneyresearch.com/project/lightfields/>  

1. Kim, C., Zimmer, H., Pritch, Y., Sorkine-Hornung, A. & Gross, M., *Scene Reconstruction from High Spatio-Angular Resolution Light Fields*. ACM Transactions on Graphics 32, 1-12 (2013).
[1]:https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20140725221413/Kim13.pdf 


## Install

 To install simply type:

 ```
 pip install DisneyDisp
 ```  


## How to use
After intallation you can use the following command line tools directly:

```
disparity
imgs2lf
lf2epi
dmap2hdf5
clif2lf
```

Just type ```command --help``` for more information. For more information on our API, please visit the [documentation](https://manusrep.github.io/DisneyDispPy/):

## License
The programm, all code and documentation is released under the  GNU General Public License Version 3.0 (GPL3). For more information see <http://www.gnu.org/licenses/>.


## Issues
Please, feel free to contribute in any way, e.g. in reporting bugs, contributing code or making sugestions for improvements. Preferable use gitHub's "Issues" and "Pull requests" but you can also contact me directly via Manuel_Tuschen@web.de.

## Release Notes

### Version 0.1

* Implemented basic file storage as .hdf5 files.
* Implemented Scene reconstruction for the center line including iterative radiance update.
* Supports multicore processing. 

# Robust-NTF
A PyTorch (CPU/GPU) implementation of Robust Non-negative Tensor Factorization (rNTF), as will appear in,

> Dey, N., et al. *Robust Non-negative Tensor Factorization, Diffeomorphic Motion Correction, and Functional Statistics to Understand Fixation in Fluorescence Microscopy*, MICCAI, October 2019.

![](https://github.com/neel-dey/robustNTF/raw/master/paper_pipeline.png)

Pre-print available [here](https://drive.google.com/file/d/1jTshyUb7B5lRtrSncXjbXVkNmqaz4kt4/view?usp=sharing).

rNTF computes a low-rank non-negative approximation of a tensor corrupted with structured sparse outliers, while allowing for a flexible modeling of noise assumptions.

This repository implements only the rNTF portion of the paper, with the other aspects (diffeomorphic motion correction for granular motion, and functional paired testing for spectra) being straightforward applications of [ANTs](http://stnava.github.io/ANTs/) and [PACE](http://www.stat.ucdavis.edu/PACE/), respectively. Please feel free to contact me at neel DOT dey AT nyu DOT edu for questions regarding any of the above.

## Features
* Implements the low-rank and sparse non-negative canonical polyadic decomposition/tensor factorization proposed in the above paper. The paper presented methodology for 3-dimensional tensors, whereas this package can handle tensors of arbitrary dimension.
	* The low-rank tensor is parameterized as the sum of individual rank-one tensors, thus obviating the need for the rank constraint.
	
	* The measure of fit between the tensor and its reconstruction is chosen to be the beta-divergence, an information-geometric divergence that allows flexible modeling of noise. For example, beta = 2, 1, and 0 correspond to Gaussian, Poisson and multiplicative Gamma noise, respectively. Values in between interpolate between assumptions.
	
	* Structured sparsity is imposed on the sparse outliers by regularizing them by the L2,1-norm.

* By using PyTorch as the backend for computations, rNTF can be performed either on the CPU or on the GPU for accelerated performance.

* Optionally, missing values can be imputed during the optimization using Expectation-Maximization. The factorization will no longer be identifiable, but works well in practice.

## Installation and dependencies
If you are using Anaconda, simply clone or download this repository, navigate to the folder, open a terminal and type:
```bash
conda env create -f environment.yml
```
That will create a conda virtual environment with all required dependencies. The main dependencies are:
  * pytorch
  * tensorly

Matplotlib, Jupyter and their dependencies are only included to run the notebook in the './example_notebooks/' folder.

This package was tested with PyTorch 1.2. with CUDA 10.0, but should be compatible with everything 0.4 and up. If any incompatibility is found, please open an issue.

## Minimal working example

```python
from robust_ntf.robust_ntf import robust_ntf

# To use the GPU at fp64 precision (use FloatTensor for fp32):
torch.set_default_tensor_type(torch.cuda.DoubleTensor)

# Generate random data:
data = torch.rand(30,40,50,60).cuda()

# Perform rNTF (main args listed for clarity):
factors, outlier, obj = robust_ntf(data, rank=2, beta=1.5, reg_val=10, tol=1e-4)
```

An example with more meaningful data is provided in the `./example_notebooks/` folder.

## Reference

If you use this package for your research, please consider citing our upcoming paper:

Dey, N., Messinger, J., Smith, R.T., Curcio, C.A., Gerig, G. "Robust Non-negative Tensor Factorization, Diffeomorphic Motion Correction, and Functional Statistics to Understand Fixation in Fluorescence Microscopy, *International Conference on Medical Image Computing and Computer-Assisted Intervention.*  Springer, 2019.

## Acknowledgments

* [TensorLy](https://github.com/tensorly/tensorly) was used for matrix to tensor conversions and vice versa.

* Several implementation ideas taken from the robust-NMF code by Fevotte & Dobigeon, available [here](http://dobigeon.perso.enseeiht.fr/applications/app_hyper_rLMM.html) and [here](https://www.irit.fr/~Cedric.Fevotte/extras/tip2015/code.zip).

* (shameless plug) My PyTorch implementation of robust-NMF, available [here](https://github.com/neel-dey/robust-nmf).

## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

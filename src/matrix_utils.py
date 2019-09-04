#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions used in computing Robust Non-negative Tensor
Factorization.

Contents
--------
    beta_divergence() : computes the beta divergence between two matrices.
    L21_norm() : returns the L_{2,1} norm of a matrix.
    kr_bcd() : Khatri-Rao product in the context of block coordinate descent
               in tensor factorizations.
    khatri_rao_product() : Khatri-Rao product of a list of matrices.

If you find bugs and/or limitations, please email neel DOT dey AT nyu DOT edu.

Created March 2019, refactored September 2019.

"""

import torch


def beta_divergence(mat1, mat2, beta):
    """Compute the beta divergence between two matrices.

    This follows the definition of the beta divergence used by Févotte, et al.
    in "Algorithms for nonnegative matrix factorization with the β-divergence."
    , Neural computation (2011). Another definition of the beta divergence used
    by Cichocki & Amari in "Families of alpha-beta-and gamma-divergences:
    Flexible and robust measures of similarities", Entropy (2010) shifts the
    values of beta by one.

    Special cases of beta:
        1. beta = 2 : Squared Euclidean Distance (Gaussian noise assumption)
        2. beta = 1 : Kullback-Leibler Divergence (Poisson noise assumption)
        3. beta = 0 : Itakura-Saito Divergence (multiplicative gamma noise
        assumption)

    Beta values in between the above interpolate between assumptions.

    NOTE: If beta = 0, the data cannot contain any zero values. If beta = 1,
    Fevotte and Dobigeon explicitly work around zero values in their version of
    the KL-divergence as shown below. beta = 2 is just the squared Frobenius
    norm of the difference between the two matrices. With the squaring, it is
    no longer an actual distance metric.

    Parameters
    ----------
    mat1, mat2 : matrices
        Matrices between which to calculate the beta divergence.

    beta : float, range [0, 2]
        Parameter of the beta divergence.

    Returns
    -------
    beta_div : float
        the beta-divergence between mat1 and mat2.

    """

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if mat1.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Inline function for vectorizing arrays for readability:
    vec = lambda X: X.flatten()

    # Main section:
    # If/else through the special limiting cases of beta, otherwise use the
    # last option:

    if beta == 2:
        # Gaussian assumption.
        beta_div = 0.5*(torch.norm(mat1 - mat2, p='fro')**2)

    elif beta == 1:
        # Poisson assumption.

        # Finding elements that would cause a division by zero/issues with log:
        zeromask = mat1 <= eps
        onemask = ~zeromask

        beta_div = (torch.sum((mat1[onemask] *
                              torch.log(mat1[onemask]/mat2[onemask])) -
                              mat1[onemask] + mat2[onemask]) +
                    torch.sum(mat2[zeromask]))

    elif beta == 0:
        # Multiplicative gamma assumption.
        beta_div = torch.sum(vec(mat1)/vec(mat2) -
                             torch.log(vec(mat1)/vec(mat2))) - len(vec(mat1))

    else:
        # General case.
        beta_div = torch.sum(vec(mat1)**beta + (beta-1)*vec(mat2)**beta
                             - beta*vec(mat1)*(vec(mat2))**(beta-1))\
                          / (beta*(beta-1))

    return beta_div


def L21_norm(mat):
    r"""Compute the L_{2,1} norm of a matrix.

    Mathematically, for a matrix M of size n times m, the L_{2,1} norm is,
    $ \| M \|_{2,1} = \sum_{i=1}^{n} \sqrt{ \sum_{j=1}^{m} M_{ij}^2} $

    Parameters
    ----------
    mat : matrix
        Matrix to get the 2,1 norm of.

    Returns
    -------
    L_{2,1} norm of input matrix.
    """

    return torch.sum(torch.sqrt(torch.sum(mat**2, dim=0)))


def kr_bcd(matrices, skip_idx):
    """Utility function for the Khatri-Rao product of a list of matrices
    in the context of block coordinate descent solving for factor matrices.

    One matrix is held constant and removed from the list of matrices, the list
    is then reversed and the Khatri-Rao product is taken.

    For mathematical intuition as to why this is done, see Kolda & Bader,
    "Tensor Decompositions and Applications", 2009; Section 3.0.

    Parameters
    ----------
    matrices : list
        List of factor matrices corresponding to rNTF.

    skip_idx : int
        index for mode of interest being solved for.

    Returns
    -------
    Khatri-Rao product relevant to block coordinate descent in a tensor
    factorization.

    """
    # Remove factor matrix being solved for:
    matrices = [matrices[i] for i in range(len(matrices)) if i != skip_idx]

    # Reverse list:
    matrices = matrices[::-1]

    return khatri_rao_product(matrices)


def khatri_rao_product(matrices):
    """Khatri-Rao (columnwise matching Kronecker) product of a list of matrices

    NOTE: This is the same implementation as TensorLy's, with the slight
    change of not allocating new arrays to save on GPU memory and with masking
    and weighing removed, and variables renamed for readability.

    Parameters
    ----------
    matrices : list
        List of matrices.

    Returns
    -------
    krp : matrix
        Khatri-Rao product of a list of matrices with the same number of
        columns.

    """

    if len(matrices) < 2:
        raise ValueError('khatri_rao_product() requires a list of at least 2'
                         'matrices, but {} given.'.format(len(matrices)))

    n_col = (matrices[0]).shape[1]

    # TensorLy's loop to calculate Khatri-Rao product:
    for idx, matrix in enumerate(matrices[1:]):
        # Initializing:
        if not idx:
            krp = matrices[0]

        # Get shapes:
        s1, s2 = krp.shape
        s3, s4 = matrix.shape

        # Error checking:
        if not s2 == s4 == n_col:
            raise ValueError('All matrices should have the'
                             'same number of columns.')

        # Calculate Khatri-Rao product:
        krp = ((krp.reshape((s1, 1, s2)) * matrix.reshape((1, s3, s4)))
               .reshape((-1, n_col)))

    return krp

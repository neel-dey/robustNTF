#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch implementation of Robust Non-negative Tensor Factorization.

See docstring for robust_ntf() for a description of overall functionality.

Contents
--------
    robust_ntf() : computes Robust Non-negative Tensor Factorization of a given
                   input tensor.
    initialize_rntf() : provides initial estimates for factor matrices and the
                        outlier tensor.
    update_factor() : multiplicatively updates the current estimate of a given
                      factor matrix.
    update_outlier() : multiplicatively updates the current estimate of a given
                       matricized outlier tensor.

If you find bugs and/or limitations, please email neel DOT dey AT nyu DOT edu.

Created March 2019, refactored September 2019.
"""

import torch
from torch.nn.functional import normalize
import tensorly as tl
from .foldings import folder, unfolder
from .matrix_utils import kr_bcd, beta_divergence, L21_norm
tl.set_backend('pytorch')


def robust_ntf(data, rank, beta, reg_val, tol, init='random', max_iter=1000,
               print_every=10, user_prov=None):
    """Robust Non-negative Tensor Factorization (rNTF)

    This function decomposes an input non-negative tensor into the sum of
    component rank-1 non-negative tensors (returned as a series of factor
    matrices), and a group-sparse non-negative outlier tensor that does not
    fit within a low-rank multi-linear model.

    The objective function is a weighted sum of the beta-divergence and L2,1
    norm, which allows for flexible noise modeling and imposing sparsity on the
    outliers. Missing values can be optionally handled via Expectation-
    Maximization. However, the model will no longer be identifiable.

    For more details, see Dey, N., et al. "Robust Non-negative Tensor
    Factorization, Diffeomorphic Motion Correction, and Functional Statistics
    to Understand Fixation in Fluorescence Microscopy", MICCAI, 2019.

    Parameters
    ----------
    data : tensor
        An n-dimensional non-negative tensor. Missing values should be NaNs.

    rank : int
        Rank of the factorization/number of components.

    beta : float, range [0, 2]
        Float parameterizing the beta divergence.
        Values at certain limits:
            beta = 2: Squared Euclidean distance (Gaussian noise assumption)
            beta = 1: Kullback-Leibler divergence (Poisson noise assumption)
            beta = 0: Itakura-Saito divergence (multiplicative gamma noise
            assumption)
        Float values in between these integers interpolate between assumptions.
        Values outside of this range contain dragons.

    reg_val : float
        Weight for the L2,1 penalty on the outlier tensor. Needs tuning
        specific to the range of the data. Start high and work your way down.

    tol : float
        tolerance on the iterative optimization.

    init : str, {'random' (default), 'user'}
        Initialization strategy.

        Valid options:
            1. 'random' : initialize all factor matrices and outlier tensor
                          uniformly at random.

            2. 'user' : you provide a dictionary containing initializations
                        for the factor matrices and outlier tensor. Must be
                        passed in the 'user_prov' paramter.

    max_iter : int
        Maximum number of iterations to compute rNTF.

    print_every : int
        Print optimization progress every 'print_every' iterations.

    user_prov : None | dict
        Only relevant if init == 'user', i.e., you provide your own
        initialization. If so, provide a dictionary with the format:
        user_prov['factors'], user_prov['outlier'].


    Returns
    -------
    matrices : list
        A list of factor matrices retrieved from the decomposition.

    outlier : tensor
        The outlier tensor retrieved from the decomposition.

    obj : array, shape (n_iterations,)
        The history of the optimization.

    """

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize rNTF:
    matrices, outlier = initialize_rntf(data, rank, init, user_prov)

    # Set up for the algorithm:
    # Initial approximation of the reconstruction:
    data_approx = matrices[0]@(kr_bcd(matrices, 0).t())

    data_approx = folder(data_approx, data, 0) + outlier + eps

    # EM step:
    ind = torch.ones(data.size())
    ind[torch.isnan(data) == 1] = 0

    data_n = data.clone()
    data_n[ind == 0] = 0
    data_imp = data_n + (1 - ind)*data_approx

    del data

    fit = torch.zeros(max_iter+1)
    obj = torch.zeros(max_iter+1)

    # Monitoring convergence:
    fit[0] = beta_divergence(data_imp, data_approx, beta)
    obj[0] = fit[0] + reg_val*L21_norm(unfolder(outlier, 0))

    # Print initial iteration:
    print('Iter = 0; Obj = {}'.format(obj[0]))
    # pdb.set_trace()

    for iter in range(max_iter):

        # EM step:
        data_imp = data_n + (1 - ind)*data_approx

        # Block coordinate descent/loop through modes:
        for mode in range(len(data_n.shape)):

            # Khatri-Rao product of the matrices being held constant:
            kr_term = kr_bcd(matrices, mode).t()

            # Update factor matrix in mode of interest:
            matrices[mode] = update_factor(unfolder(data_imp, mode),
                                           unfolder(data_approx, mode),
                                           beta,
                                           matrices[mode],
                                           kr_term)

            # Update reconstruction:
            data_approx = (folder(matrices[mode]@kr_term, data_n, mode)
                           + outlier
                           + eps)

            # Update outlier tensor:
            outlier = (folder(update_outlier(unfolder(data_imp, mode),
                                             unfolder(data_approx, mode),
                                             unfolder(outlier, mode),
                                             beta, reg_val),
                              data_n, mode))

            # Update reconstruction:
            data_approx = (folder(matrices[mode]@kr_term, data_n, mode)
                           + outlier
                           + eps)

        # Monitor optimization:
        fit[iter+1] = beta_divergence(unfolder(data_imp, 0),
                                      unfolder(data_approx, 0),
                                      beta)
        obj[iter+1] = fit[iter+1] + reg_val*L21_norm(unfolder(outlier, 0))

        if iter % print_every == 0:  # print progress
            print('Iter = {}; Obj = {}; Err = {}'.format(iter+1, obj[iter+1],
                  torch.abs((obj[iter]-obj[iter+1])/obj[iter])))

        # Termination criterion:
        if torch.abs((obj[iter]-obj[iter+1])/obj[iter]) <= tol:
            print('Algorithm converged as per defined tolerance')
            break

        if iter == (max_iter - 1):
            print('Maximum number of iterations achieved')

    # In case the algorithm terminated early:
    obj = obj[:iter]
    fit = fit[:iter]

    return matrices, outlier, obj


def initialize_rntf(data, rank, alg, user_prov=None):
    """Intialize Robust Non-negative Tensor Factorization.

    This function retrieves an initial estimate of factor matrices and an
    outlier tensor to intialize rNTF with.

    Parameters
    ----------
    data : matrix
        A matricized version of the input non-negative tensor.

    rank : int
        Rank of the factorization/number of components.

    alg : str, {'random' (default), 'user'}
        Initialization strategy.

        Valid options:
            1. 'random' : initialize all factor matrices and outlier tensor
                          uniformly at random.

            2. 'user' : you provide a dictionary containing initializations
                        for the factor matrices and outlier tensor. Must be
                        passed in the 'user_prov' parameter.

    user_prov : None | dict
        Only relevant if init == 'user', i.e., you provide your own
        initialization. If so, provide a dictionary with the format:
        user_prov['factors'], user_prov['outlier'].

    Returns
    -------
    matrices : list
        List of the initial factor matrices.

    outlier : tensor
        Intial estimate of the outlier tensor.

    """

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Initialize outliers with uniform random values:
    outlier = (torch.rand(data.size()) + eps)

    # Initialize basis and coefficients:
    if alg == 'random':
        print('Initializing rNTF with uniform noise.')

        matrices = list()
        for idx in range(len(data.shape)):
            matrices.append(torch.rand(data.shape[idx], rank) + eps)

        return matrices, outlier

    elif alg == 'user':
        print('Initializing rNTF with user input.')

        matrices = user_prov['factors']
        outlier = user_prov['outlier']

        return matrices, outlier

    else:
        # Making sure the user doesn't do something unexpected:
        # Inspired by how sklearn deals with this:
        raise ValueError(
            'Invalid algorithm (typo?): got %r instead of one of %r' %
            (alg, ('random', 'nndsvdar', 'user')))


def update_factor(data, data_approx, beta, factor, krp):
    """Update factor matrix.

    Implements the factor matrix update for robust non-negative tensor
    factorization.

    Parameters
    ----------
    data : matrix
        Matricized tensor in the mode corresponding to the factor matrix being
        solved for.

    data_approx : matrix
        Matricized version of the low-rank + sparse tensor reconstruction from
        the factor matrices, in the mode correspondidng to the factor matrix
        being solved for.

    beta : float, range [0, 2]
        Parameterization of the beta divergence. See docstring of robust_ntf()
        for details.

    factor : matrix
        Current estimate of the factor matrix being solved for.

    krp : matrix
        Current estimate of the Khatri-Rao product of the factor matrices
        currently being held constant while estimating 'factor', for block
        coordinate descent.

    Returns
    -------
    Multiplicative update for the factor matrix of interest.
    """

    return factor * ((data*(data_approx**(beta-2))@krp.t()) /
                     ((data_approx**(beta-1))@krp.t()))


def update_outlier(data, data_approx, outlier, beta, reg_val):
    """Update matricized outlier tensor.

    Implements the matricized outlier matrix update for robust non-negative
    tensor factorization.

    Parameters
    ----------
    data : matrix
        Matricized input tensor in the mode corresponding to the matricization
        of the outlier tensor.

    data_approx : matrix
        Matricized version of the low-rank tensor reconstruction from the
        factor matrices, in the mode correspondidng to the factor matrix being
        solved for.

    outlier : matrix
        Current estimate of the matricized outlier tensor being solved for.

    beta : float, range [0, 2]
        Parameterization of the beta divergence. See docstring of robust_ntf()
        for details.

    reg_val : float
        Weight for the L2,1 penalty on the outlier tensor. Needs tuning
        specific to the range of the data. Start high and work your way down.

    Returns
    -------
    Multiplicative update for the matricized outlier tensor.
    """

    # Utilities:
    # Defining epsilon to protect against division by zero:
    if data.type() == 'torch.cuda.FloatTensor':
        eps = 1.3e-7  # Slightly higher than actual epsilon in fp32
    else:
        eps = 2.3e-16  # Slightly higher than actual epsilon in fp64

    # Using inline functions for readability:
    bet1 = lambda X: X**(beta-1)
    bet2 = lambda X: X**(beta-2)

    return outlier * ((data*bet2(data_approx)) / (bet1(data_approx) +
                                                  reg_val*normalize(outlier,
                                                                    p=2,
                                                                    dim=0,
                                                                    eps=eps)))

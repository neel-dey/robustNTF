#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains functions to matricize (unfold) and tensorize (fold).

This mainly exists to undo the indexing style of PyTorch/TensorLy and revert to
the indexing style of Kolda and Bader, without having to rewrite several
functions in both PyTorch and TensorLy.

For background, PyTorch and TensorLy both assume C-style indexing, whereas I
originally developed my code based on Kolda & Bader's formulation which uses
Fortran-style indexing, which is slower. If you would like to rewrite rNTF to
use the faster C-style indexing, please feel free to open a PR.

An excellent blog post about this: http://jeankossaifi.com/blog/unfolding.html

Contents
--------
    permutation_generator() : generates the permutation order for the tensor
                              such that the correct operation will be applied
                              by tl.fold() and tl.unfold().
    folder() : tensorizes a matrix using the permutation order given by
               permutation_generator() and tl.fold().
    unfolder() : matricizes a tensor using the permutation order given by
                 permutation_generator() and tl.unfold().

If you find bugs and/or limitations, please email neel DOT dey AT nyu DOT edu.

Created March 2019, refactored September 2019.
"""

import tensorly as tl
tl.set_backend('pytorch')


def permutation_generator(dims, mode):
    """Generates the permutation order for the tensor such that the correct
    operation will be applied by tl.fold() and tl.unfold() which use a
    different style of indexing than the rest of rNTF.

    Parameters
    ----------
    dims : tensor shape
        Shape of original tensor.
    mode : int
        Mode to perform folding/unfolding on.

    Returns
    -------
    ordering : list
        Permutation order for the tensor for correct operations.

    """

    ordering = list(range(len(dims)))
    ordering.remove(mode)
    ordering.reverse()
    ordering.insert(mode, mode)

    return ordering


def folder(mat, ten, mode):
    """Tensorizes a matrix using the permutation order given by
    permutation_generator() and then by using tl.fold().

    Parameters
    ----------
    mat : matrix
        Matrix to be rearranged into a tensor.
    ten : tensor
        Original data tensor, used here for a reference shape.
    mode : int
        Mode along which to rearrange.

    Returns
    -------
    Tensorization of the input matrix.

    """

    ten_shape = ten.size()
    permute_order = permutation_generator(ten_shape, mode)

    # Ugly one-liner used so as to not allocate intermediates and save memory.
    return (tl.fold(mat, mode, ten.permute(permute_order).size())
            .permute(permute_order))


def unfolder(ten, mode):
    """Matricizes a tensor using the permutation order given by
    permutation_generator() and then by using tl.unfold().

    Parameters
    ----------
    ten : tensor
        Tensor to be matricized.
    mode : int
        Mode along which to rearrange.

    Returns
    -------
    Matricization of the input tensor.

    """
    ten_shape = ten.size()
    permute_order = permutation_generator(ten_shape, mode)

    return tl.unfold(ten.permute(permute_order), mode)

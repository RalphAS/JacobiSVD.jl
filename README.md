# JacobiSVD

![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Build Status](https://travis-ci.com/RalphAS/JacobiSVD.jl.svg?branch=master)](https://travis-ci.com/RalphAS/JacobiSVD.jl)
[![codecov.io](http://codecov.io/github/RalphAS/JacobiSVD.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/JacobiSVD.jl?branch=master)

## Introduction

JacobiSVD is a Julia package wrapping the LAPACK subroutines for computing
the singular value decomposition (SVD) of general dense matrices using
Jacobi algorithms. The advantage of the Jacobi scheme is the higher
accuracy of the computed singular values (compared to QR and divide-and-conquer
algorithms). The disadvantage is that they are relatively inefficient unless
one combines them with elaborate preconditioning. LAPACK includes
routines which organize such preconditioning.

Caveat: extra accuracy is guaranteed only for matrices which can be
expressed as `A = B * D` where `B` is well-conditioned and `D` is a diagonal
scaling. It also obtains for forms `A = D1 * C * D2` where `C` is
well-conditioned and `D1, D2` are diagonal scalings, but theory is lacking.
For general matrices, results may be no better than standard methods.

## Usage

This package provides a `LinearAlgebra`-style function `jsvd!`.
```julia
S = jsvd!(A)
```
which returns an `SVD` object, like `svd!(A)`, but perhaps more accurate.
It also exports low-level wrappers `gesvj!` and `gejsv!`; see the docstrings
and LAPACK documentation for details.

## References

Z. Drmac and K. Veselic: New fast and accurate Jacobi SVD algorithm I.
SIAM J. Matrix Anal. Appl. Vol. 35, No. 2 (2008), pp. 1322-1342.
[LAPACK Working note 169](http://www.netlib.org/lapack/lawnspdf/lawn169.pdf)

Z. Drmac and K. Veselic: New fast and accurate Jacobi SVD algorithm II.
SIAM J. Matrix Anal. Appl. Vol. 35, No. 2 (2008), pp. 1343-1362.
[LAPACK Working note 170](http://www.netlib.org/lapack/lawnspdf/lawn170.pdf)

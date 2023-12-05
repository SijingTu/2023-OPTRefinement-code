# Finding Denser Subgraphs and Larger Cuts with Local Improvements

This repository contains the code used in our paper titled "Finding Denser Subgraphs and Larger Cuts with Local Improvements". The algorithms are designed to find larger graph cuts and denser subgraphs through a local improvement framework.

## Table of Contents

- [Finding Denser Subgraphs and Larger Cuts with Local Improvements](#finding-denser-subgraphs-and-larger-cuts-with-local-improvements)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Usage](#usage)
    - [ds-with-small-changes](#ds-with-small-changes)
    - [cut-with-small-changes](#cut-with-small-changes)

## Overview

This repository contains the codes for the following two problems: 


1. Discovering denser subgraphs with local modifications (`ds-with-small-changes`).
2. Finding larger cuts with local modifications. (`cut-with-small-changes`).

Some methods are based on the semidefinite programming approach, and make use of the Mosek optimization library.

## Installation

To use the code, you'll need a valid license for [Mosek](https://www.mosek.com/), which is used for solving the semidefinite programs in our algorithms.

## Datasets

The repository we demonstrate how to run our codes in synthetic datasets. Running on other datasets are possible with minor changes in the initialization steps in the code.

## Usage

### ds-with-small-changes

```bash
python solver_small.py $k 'move_out' 'sb_model_sparse' 'k' $sigma
```


- `$k` is a parameter which represents the number of nodes selected.
- 'move_out' means the algorithm first move $k$ nodes out, it can be replaced with 'not_move_out'.
- 'sb_model_sparse' is a dataset we use, it can be replaced with the name of your graph dataset, for example 'sb_model_dense'.
- 'k' means `$k` select $k$ nodes. It can be replaced with 'ratio', in this case `$k` select $k\%n_0$ nodes. 
- `$sigma` represents the $\sigma$-quasi-elimination-order, we select $5$ for the experiments;

### cut-with-small-changes

The following code is a demostration. 

```bash
python solver_small.py $k 'sb_model_sparse'
```

- `$k` is a parameter which represents the number of nodes selected.
- 'sb_model_sparse' is a dataset we use, it can be replaced with the name of your graph dataset, for example 'sb_model_dense'.


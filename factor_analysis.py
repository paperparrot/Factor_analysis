__author__ = 'sebastien.genty'
__version__ = '0.5'

import numpy as np
import pandas as pd

"""
This program does factor analysis as previously done in SPSS. Dimension reduction done by PCA, followed by a Varimax
rotation to make the loadings more readable.

Docstring to be completed.
"""


def varimax_rotation(matrix, eps=1e-6, itermax=1000):
    """
    Does an orthogonal rotation of the original matrix using the Varimax method. This implementation could be changed
    to include other rotations by changing the gamma.
    :param matrix: a Numpy matrix
    :param itermax: Integer, maximum number of iterations
    :return: A numpy matrix that is the rotated form of the input matrix
    """

    # Defining the gamma for the rotation. This is the section that can be changed (in comments). Gamma would need to be
    # included
    gamma = 1.0
    # if gamma == None:
    #     if (method == 'varimax'):
    #         gamma = 1.0
    #     if (method == 'quartimax':
    #         gamma = 0.0

    nrow, ncol = matrix.shape
    rotated_matrix = np.eye(ncol)
    temp_var = 0

    for i in range(itermax):
        lam_rot = np.dot(matrix, rotated_matrix)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(matrix.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        rotated_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < temp_var * (1 + eps):
            break
        temp_var = var_new

    return rotated_matrix


def pca(dataframe, var_x, var_y, stop=-1, rotation='varimax'):
    """
    Principal Component Analysis implementation using the correlation matrix. This could be expanded to use the
    covariance matrix by adding an if statement when the correlation matrix is created.
    :param dataframe: Entire datafile. Session ID should be the index
    :param var_x: First variable of the range to be included in the analysis
    :param var_y: Last variable of the range to be included in the analysis
    :param stop: Criteria to stop. Integer number indicating the number of factors to be included.
    :param rotation:
    :return:
    """
    # Load the inital data frame
    initial_df = dataframe.ix[:, var_x: var_y].copy()

    # Computing the correlation matrix, finding the eigenvalues and eigenvectors, then sorting them
    corr_matrix = np.corrcoef(initial_df, rowvar=0)
    eigenvals, eigenvects = np.linalg.eig(corr_matrix)
    eigen_df = pd.DataFrame({'eigen_value': eigenvals,
                             'eigen_vector': eigenvects})

    eigen_df = eigen_df.sort(['eigen_value'])

    # Determines the number of components to include
    if stop > 0:
        eigen_df = eigen_df[:stop]

    else:
        eigen_df = eigen_df[eigen_df['eigen_value'] >= 1]

    # Taking the eigenvalues, putting them in a matrix along the diagonal and taking the square root
    eigenval_matrix = np.diag(eigen_df['eigen_value'])
    eigenval_matrix = np.sqrt(eigenval_matrix)

    # Calculating the loadings by multiplying the eigenvectors to the new eigenvalue matrix
    loadings_matrix = np.dot(eigen_df['eigen_value'], eigenval_matrix)

    # Performing the Varimax rotatiom
    if rotation == 'varimax':
        loadings = pd.DataFrame(varimax_rotation(loadings_matrix), index=initial_df.columns)
    else:
        loadings = pd.DataFrame(loadings_matrix, index=initial_df.columns)

    print loadings
    return loadings

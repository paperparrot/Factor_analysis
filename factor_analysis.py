__author__ = 'sebastien.genty'
__version__ = '0.7'
__status__ = 'development'

import numpy as np
import pandas as pd

"""
This program does factor analysis as previously done in SPSS. Dimension reduction done by PCA, followed by a Varimax
rotation & Kaizer normalization to make the loadings more readable.

Docstring to be completed.
"""


def communalities(matrix):
    """
    This function takes a matrix (expected to the the loadings matrix from the PCA function) and computes the
    communalities.
    :param matrix: numpy matrix or pandas DataFrame
    :return: Pandas series of the communalities
    """

    df = pd.DataFrame(matrix.square())
    output = pd.Series()

    for i in df.index():
        output[i] = df.ix[i, :].sum()

    return output


def varimax_rotation(matrix, eps=1e-6, itermax=1000):
    """
    Does an orthogonal rotation of the original matrix using the Varimax method. This implementation could be changed
    to include other rotations by changing the gamma.
    :param matrix: a Numpy matrix
    :param itermax: Integer, maximum number of iterations
    :return: A numpy matrix that is the rotated form of the input matrix
    """

    # Defining the gamma for the rotation. This is the section that can be changed (in comments). Gamma would need to be
    # included as one of the arguments.
    gamma = 1.0
    # if gamma == None:
    #     if (method == 'varimax'):
    #         gamma = 1.0
    #     if (method == 'quartimax':
    #         gamma = 0.0

    nrow, ncol = matrix.shape
    rotated_matrix = np.eye(ncol) 
    temp_var = 0

    # Need to insert part where initial matrix is multiplied by the square of the communalities
    commun = np.diag(communalities(matrix))
    matrix = np.dot(matrix, commun)

    for i in range(itermax):
        lam_rot = np.dot(matrix, rotated_matrix)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(matrix.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        rotated_matrix = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < temp_var * (1 + eps):
            break
        temp_var = var_new
        output_matrix = np.dot(matrix, rotated_matrix)

    # Need to insert part where output matrix is multiplied by the square of the communalities. Also check I am not
    # missing a step here.
    output_matrix = np.dot(output_matrix, commun)

    return output_matrix


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
    initial_df = dataframe.loc[:, var_x: var_y].copy()

    # Computing the correlation matrix, finding the eigenvalues and eigenvectors, then sorting them
    corr_matrix = np.corrcoef(initial_df, rowvar=0)
    eigenvals, eigenvects = np.linalg.eig(corr_matrix)

    return_list = list()
    
    # Transposing the eigenvectors, then creating a DataFrame that contains the Eigenvalues and vectors to sort.
    eigenvects = np.transpose(eigenvects)
    
    for val, vec in zip(eigenvals, eigenvects):
        local_dict = dict()
        local_dict['eigen_val'] = val
        local_dict['eigen_vec'] = vec
        return_list.append(local_dict)   
    
    eigen_df = pd.DataFrame(return_list)
    
    eigen_df = eigen_df.sort(['eigen_val'], ascending=False)

    # Determines the number of components to include
    if stop > 0:
        eigen_df = eigen_df[:stop]

    else:
        eigen_df = eigen_df[eigen_df['eigen_val'] >= 1]

    # Taking the eigenvalues, putting them in a matrix along the diagonal and taking the square root
    eigenval_matrix = np.diag(eigen_df['eigen_val'])
    eigenval_matrix = np.sqrt(eigenval_matrix)

    # Calculating the loadings by multiplying the eigenvectors to the new eigenvalue matrix
    loadings_matrix = np.dot(eigen_df['eigen_vec'], eigenval_matrix)
    
    # Putting the loadings into a DataFrame, and formating as necessary.
    loadings_df = pd.DataFrame()
    for i in np.arange(len(loadings_matrix)):
        temp_name = 'Factor ' + str(i)
        loadings_df[temp_name] = pd.Series(loadings_matrix[i])        
    
    # Performing the Varimax rotatiom
    if rotation == 'varimax':
        rotated = varimax_rotation(loadings_df)
        print rotated
        loadings = pd.DataFrame(rotated, index=initial_df.columns)
    else:
        loadings = loadings_df.set_index(initial_df.columns)

    return loadings

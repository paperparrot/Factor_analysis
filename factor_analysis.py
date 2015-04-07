__author__ = 'sebastien.genty'
__version__ = '0.1'

import numpy as np
import pandas as pd

"""
This program does factor analysis as previously done in SPSS. Dimension reduction done by PCA, followed by a Varimax
rotation to make the loadings more readable.

Docstring to be completed.
"""


def pca(dataframe, var_x, var_y, stop='eigen'):

    # Load the inital data frame
    initial_df = dataframe[var_x, var_y].copy()

    # Computing the correlation matrix, finding the eigenvalues and eigenvectors
    corr_matrix = np.corrcoef(initial_df, rowvar=0)
    eigenvals, eigenvects = np.linalg.eig(corr_matrix)

    # Do the formula
    loadings = pd.DataFrame(np.dot(eigenvects, np.diag(eigenvals)), index=initial_df.columns)

#!/usr/bin/env python
# coding: utf-8

import numpy as np

# Initialize the class, load data with load_data(), and generate the transformed data points via transform_data()
class PCA():
    
    def __init__(self, components=2, variance_thresh = 0.95):
        
        self.components = components
        self.variance_thresh = variance_thresh
        
    def load_data(self, np_array):
        
        self.data = np_array
        
        if self.data.dtype == 'object':
            self.data = self.data.astype('float')
        
    def get_principal_components(self):
        
        feature_means = np.mean(self.data, axis=0, keepdims=True)
        self.centered_features = self.data - feature_means
        
        cov_matrix = np.cov(self.centered_features, rowvar=False)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        cumulative = 0
        stop_idx = 0
        
        for i, eig_val in enumerate(eigenvalues):
            if (cumulative > self.variance_thresh) or (i >= self.components):
                break
                
            shared_variance = np.abs(eig_val/np.sum(eigenvalues))
            cumulative = cumulative + shared_variance
            stop_idx = i

        self.principal_components = eigenvectors[:, :(stop_idx + 1)]
        
    def transform_data(self):

        self.get_principal_components()

        self.transformed_data = np.dot(self.centered_features, self.principal_components)
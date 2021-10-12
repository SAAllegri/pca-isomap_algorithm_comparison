#!/usr/bin/env python
# coding: utf-8

import numpy as np

from scipy import spatial, sparse

from sklearn.utils import graph_shortest_path
from sklearn.neighbors import kneighbors_graph

# Initialize the class, load the data with load_data(), and obtain the z-matrix through generate_low_dimensional_representation()
class ISOMAP():
    
    def __init__(self, dimensions=2, neighbor_min=100, epsilon_thresh=0.001, dist_measure='euclidean'):
        
        self.neighbor_min = neighbor_min
        self.epsilon_thresh = epsilon_thresh
        self.dist_measure = dist_measure
        self.dimensions = dimensions
        
    def load_data(self, np_array):
        
        self.data = np_array
        self.shape = np_array.shape
        
    def generate_adjacency_matrix(self):
        
        self.distance_matrix = spatial.distance.cdist(self.data, self.data, self.dist_measure)
        distance_matrix_temporary = self.distance_matrix.copy()
        
        iso_max = np.max(distance_matrix_temporary)
        iso_min = np.min(distance_matrix_temporary)

        epsilon = ((iso_max - iso_min) / 2) + (((iso_max - iso_min) / 2) / 2)
        prev_epsilon = ((iso_max - iso_min) / 2) - (((iso_max - iso_min) / 2) / 2)

        while (np.abs((epsilon - prev_epsilon) / prev_epsilon) > self.epsilon_thresh):
            switch = np.inf
            for idx in range(self.data.shape[0]):
                edge_count = len(distance_matrix_temporary[idx][distance_matrix_temporary[idx] < epsilon])
                if edge_count < switch:
                    switch = edge_count

            prev_epsilon_local = epsilon

            # +1 to include self
            if switch < (self.neighbor_min + 1):
                epsilon = epsilon + (np.abs(epsilon - prev_epsilon) / 2)
                prev_epsilon = prev_epsilon_local
            else: 
                self.epsilon_outer = epsilon
                epsilon = epsilon - (np.abs(epsilon - prev_epsilon) / 2)
                prev_epsilon = prev_epsilon_local
                
        for idx in range(self.data.shape[0]):
            distance_matrix_temporary[idx][idx] = 0
            distance_matrix_temporary[idx][np.where(distance_matrix_temporary[idx] > self.epsilon_outer)] = 0
            
        self.adjacency_matrix = distance_matrix_temporary
        
    def generate_shortest_path_matrix(self):
        
        self.shortest_path = sparse.csgraph.dijkstra(sparse.csr_matrix(self.adjacency_matrix))
        
    def generate_low_dimensional_representation(self):
        
        self.generate_adjacency_matrix()
        self.generate_shortest_path_matrix()
        
        h = np.identity(self.shape[0]) - ((1 / self.shape[0]) * np.ones((self.shape[0], self.shape[0])))
        
        self.c_matrix = (-1 / 2) * h.dot(self.shortest_path ** 2).dot(h)
        
        eigenvalues, eigenvectors = np.linalg.eig(self.c_matrix)
        
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx].real
        eigenvectors = eigenvectors[:, idx].real
        
        reduced_eigenvalues = eigenvalues[:self.dimensions] 
        reduced_eigenvectors = eigenvectors[:, :self.dimensions]
        
        self.z_matrix = reduced_eigenvectors.dot(np.diag(reduced_eigenvalues ** (-1 / 2)))
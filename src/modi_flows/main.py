"""
MODI -- https://github.com/aleable/MODI
Contributors:
    Alessandro Lonardi
    Diego Baptista
    Caterina De Bacco
"""

import numpy as np
import networkx as nx
import sys
import warnings

from skimage.color import rgb2gray
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter

#from . import initialization as init
#from . import dynamics as dyn

import modi_flows.initialization as init
import modi_flows.dynamics as dyn 

# warn options
if not sys.warnoptions:
    warnings.simplefilter("ignore")

#####################################
############### MAIN ###############
#####################################

class MODI:

    def __init__(self, x, y, c, dt=0.3, beta=1.0, alpha=0.5, t=1.0,
                 seed=0, tol=1e-3, time_tol=1e3, verbose=False):

        self.x = x  # data 1
        self.y = y  # data 2
        self.c = c  # ground metric
        self.t = t  # sparsification threshold
        self.alpha = alpha  # penalty for unbalanced OT
        self.pflux = beta   # regularization parameter
        self.time_step = dt     # time step

        self.seed = seed    # seed spsolve
        self.time_tol = time_tol    # safety stoppage time
        self.tol = tol      # tolerance for convergence

        self.verbose = verbose

        self.r = np.zeros(1)   # swapped histogram from unbalanced OT
        self.s = np.zeros(1)   # swapped histogram from unbalanced OT
        self.g = nx.Graph()    # transport graph topology
        self.length = np.zeros(1)   # cost of edges
        self.B = np.zeros(1)   # incidence matrix (sparse matrix)
        self.forcing = np.zeros(1)  # right hand side dynamical system
        self.tdens = np.zeros(1)    # conductivities

    def exec(self):
        """
        Building the OT problem and solving it integrating the dynamics

        Return:
            J, float: optimal cost
            self.r, np.array: extended "right" distribution (accounting for auxiliary nodes)
            self.s, np.array: extended "left" distribution (accounting for auxiliary nodes)
            self.c, np.array: extended ground distance matrix, C
        """

        init.ot_setup(self)
        j = dyn.ot_solve(self)

        return j, self.r, self.s, self.c

    def setup_only(self):
        """
        To run if one wants to get the OT network, G, and H, without running the dynamics

        Return:
            self.r, np.array: extended "right" distribution (accounting for auxiliary nodes)
            self.s, np.array: extended "left" distribution (accounting for auxiliary nodes)
            self.c, np.array: extended ground distance matrix, C
        """

        init.ot_setup(self)

        return self.r, self.s, self.c


#####################################
###### FROM IMAGES TO OT SETUP ######
#####################################

def preprocessing(img1, img2, otp_alg, pooling_size, sigma, theta):
    """
    Preprocessing images to get ground metric and histograms

    Parameters:
        img1, np.array: RGB image1
        img2, np.array: RGB image2
        otp_alg, string: type of algorithm used {"multicom" or "unicom"}
        pooling_size, float: average pooling kernel size
        sigma, float: standard deviation of the Gaussian kernel
        theta, float: weight of convex combination between costs: C = (1-theta)*Y + theta*X

    Return:
        C, np.array: ground matrix
        g, np.array: histogram1/tensor1 passed to the OT setup
        h, np.array: histogram2/tensor2 passed to the OT setup
    """

    def resizing_pooling(img1, img2, pooling_size):
        """
        Resizing and average pooling

        Parameters:
            img1, np.array: RGB image1
            img2, np.array: RGB image2
            pooling_size, float: average pooling kernel size

        Return:
            img1_pool, np.array: RGB pooled image1
            img2_pool, np.array: RGB pooled image2
        """

        # closest integer divisible by size of the mask
        h1 = int(img1.shape[0] - (img1.shape[0] % pooling_size))
        w1 = int(img1.shape[1] - (img1.shape[1] % pooling_size))
        h2 = int(img2.shape[0] - (img2.shape[0] % pooling_size))
        w2 = int(img2.shape[1] - (img2.shape[1] % pooling_size))

        img1_pool = block_reduce(img1[:h1, :w1, :], block_size=(pooling_size, pooling_size, 1), func=np.mean)
        img2_pool = block_reduce(img2[:h2, :w2, :], block_size=(pooling_size, pooling_size, 1), func=np.mean)

        return img1_pool, img2_pool

    def to_grayscale(img1, img2):
        """
        Grayscale images, conversion calibrated for contemporary CRT phosphors:
            Y = 0.2125 R + 0.7154 G + 0.0721 B

        Parameters:
            img1, np.array: RGB image1
            img2, np.array: RGB image2

        Return:
            img1_gs, np.array: image1 in gray scale
            img2_gs, np.array: image2 in g ray scale
        """

        # grayscale intensities
        img1_gs = rgb2gray(img1)
        img2_gs = rgb2gray(img2)

        return img1_gs, img2_gs

    def histograms_computation(img1, img2, otp_alg):
        """
        Compute histograms for OT problem setup, given a pair of images

        Parameters:
            img1, np.array: RGB image1
            img2, np.array: RGB image2
            otp_alg, string: type of algorithm used

        Return:
            g, np.array: histogram1/tensor1 passed to the OT setup
            h, np.array: histogram2/tensor2 passed to the OT setup
        """

        if otp_alg == "multicom":
            g_1 = np.reshape(img1[:,:,0], (img1[:,:,0].shape[0]*img1[:,:,0].shape[1], 1), order='C')
            g_2 = np.reshape(img1[:,:,1], (img1[:,:,1].shape[0]*img1[:,:,1].shape[1], 1), order='C')
            g_3 = np.reshape(img1[:,:,2], (img1[:,:,2].shape[0]*img1[:,:,2].shape[1], 1), order='C')
            h_1 = np.reshape(img2[:,:,0], (img2[:,:,0].shape[0]*img2[:,:,0].shape[1], 1), order='C')
            h_2 = np.reshape(img2[:,:,1], (img2[:,:,1].shape[0]*img2[:,:,1].shape[1], 1), order='C')
            h_3 = np.reshape(img2[:,:,2], (img2[:,:,2].shape[0]*img2[:,:,2].shape[1], 1), order='C')

            g = np.hstack((g_1, g_2, g_3))
            h = np.hstack((h_1, h_2, h_3))

        else:
            g = np.reshape(img1, (img1.shape[0]*img1.shape[1], 1), order='C')
            h = np.reshape(img2, (img2.shape[0]*img2.shape[1], 1), order='C')
            g = g.transpose()[0]
            h = h.transpose()[0]

        return g, h

    def euclidean_ground_distance(img1_gs, img2_gs):
        """
        Computing Euclidean distance between pixels

        Parameters:
            img1_gs, np.array: image1 in gray scale
            img2_gs, np.array: image2 in gray scale

        Return:
            c1, np.array: contribution to cost given by Euclidian distance of pixels
        """

        max_dim1 = max(img1_gs.shape[0], img1_gs.shape[1]) - 1
        max_dim2 = max(img2_gs.shape[0], img2_gs.shape[1]) - 1

        index_matrix_1 = np.zeros((img1_gs.shape[0], img1_gs.shape[1]), dtype=object)
        index_matrix_2 = np.zeros((img2_gs.shape[0], img2_gs.shape[1]), dtype=object)
        for i in range(img1_gs.shape[0]):
            for j in range(img1_gs.shape[1]):
                index_matrix_1[i, j] = (i/max_dim1, j/max_dim1)
        for i in range(img2_gs.shape[0]):
            for j in range(img2_gs.shape[1]):
                index_matrix_2[i, j] = (i/max_dim2, j/max_dim2)

        # flattening
        index_flat_1 = np.reshape(index_matrix_1, (index_matrix_1.shape[0]*index_matrix_1.shape[1], 1), order='C').T[0]
        index_flat_2 = np.reshape(index_matrix_2, (index_matrix_2.shape[0]*index_matrix_2.shape[1], 1), order='C').T[0]

        c1 = np.zeros((len(index_flat_1), len(index_flat_2)))
        for i in range(len(index_flat_1)):
            for j in range(len(index_flat_2)):
                c1[i, j] = ((index_flat_1[i][0]-index_flat_2[j][0])**2 + (index_flat_1[i][1]-index_flat_2[j][1])**2)**0.5

        c1 = c1/np.max(c1)    # maximum distance equal to 1

        return c1

    def color_ground_distance(img1_gs, img2_gs, sigma):
        """
        Computing color distance between pixels

        Parameters:
            img1_gs, np.array: image1 in gray scale
            img2_gs, np.array: image2 in gray scale
            sigma, float: standard deviation of the Gaussian kernel

        Return:
            c2, np.array: contribution to cost given by l1 distance of pixel intensities
        """

        img1_gs_gauss = gaussian_filter(img1_gs, sigma)
        img2_gs_gauss = gaussian_filter(img2_gs, sigma)
        flattened_gauss_1 = np.reshape(img1_gs_gauss, (img1_gs_gauss.shape[0]*img1_gs_gauss.shape[1], 1), order='C').T[0]
        flattened_gauss_2 = np.reshape(img2_gs_gauss, (img2_gs_gauss.shape[0]*img2_gs_gauss.shape[1], 1), order='C').T[0]

        c2 = np.zeros((len(flattened_gauss_1), len(flattened_gauss_2)))
        for i in range(len(flattened_gauss_1)):
            for j in range(len(flattened_gauss_2)):
                c2[i, j] = abs(flattened_gauss_1[i] - flattened_gauss_2[j])

        c2 = c2/np.max(c2)   # maximum distance equal to 1

        return c2

    if otp_alg == "multicom":

        img1, img2 = resizing_pooling(img1, img2, pooling_size)
        img1_gs, img2_gs = to_grayscale(img1, img2)
        g, h = histograms_computation(img1, img2, otp_alg)
        c1 = euclidean_ground_distance(img1_gs, img2_gs)
        c2 = color_ground_distance(img1_gs, img2_gs, sigma)

    else:
        img1, img2 = resizing_pooling(img1, img2, pooling_size)
        img1_gs, img2_gs = to_grayscale(img1, img2)
        g, h = histograms_computation(img1_gs, img2_gs, otp_alg)
        c1 = euclidean_ground_distance(img1_gs, img2_gs)
        c2 = color_ground_distance(img1_gs, img2_gs, sigma)

    # convex combination of color cost and pixels' position
    C = (1-theta)*c1 + theta*c2

    # safety parameter to avoid entries of C equal to 0
    eps = 1e-5
    C = C + eps

    return C, g, h

"""
MODI -- https://github.com/aleable/MODI
Contributors:
    Alessandro Lonardi
    Diego Baptista
    Caterina De Bacco
"""

import numpy as np
import networkx as nx


def ot_setup(self):
    """
    Construct the OT problem
    """

    def sparsunb(c, x, y):
        """
        Sparsifying ground distance and adding nodes for unbalanced OT
        Entries of the ground metric cost that are set to -1 correspond to edges to trim

        Parameters:
            c, np.array: ground distance matrix, C
            x, np.array: original histogram/tensor 1
            y, np.array: original histogram/tensor 2

        Return:
            c, np.array: extended ground distance matrix (accounting for auxiliary nodes)
            r, np.array: extended "right" distribution (accounting for auxiliary nodes)
            s, np.array: extended "left" distribution (accounting for auxiliary nodes)
        """

        # multicommodity RGB case
        try:
            if x.shape[1] == 3 and y.shape[1] == 3:

                # sparse ground metric
                c[c >= self.t] = -1

                # Unbalanced OT
                # Swapping histograms: biggest mass -> s (on the "right")
                lambda_g = np.sum(x)
                lambda_h = np.sum(y)
                if lambda_g >= lambda_h:
                    s = x
                    r = y
                    c = c.transpose()  # swapped histograms
                else:
                    s = y
                    r = x

                # Sparse matrix: add transhipment links
                # extended histogram with transhipment vertex
                r = np.vstack((r, np.array([0, 0, 0])))
                s = np.vstack((s, np.array([0, 0, 0])))

                # Instead of one with cost = threshold, 2 with cost thresh/2, so that mass does not come back
                c = np.vstack([c, np.ones(s.shape[0] - 1) * self.t * 0.5])
                col = np.append(np.ones(r.shape[0] - 1) * self.t * 0.5, -1)  # !
                c = np.column_stack((c, col))

                # extended histogram for unbalanced OT
                r = np.vstack((r, np.array([np.sum(s[:, 0]) - np.sum(r[:, 0]),
                                            np.sum(s[:, 1]) - np.sum(r[:, 1]),
                                            np.sum(s[:, 2]) - np.sum(r[:, 2])])))
                s = np.vstack((s, np.array([0, 0, 0])))

                # adding row and col to cost matrix for unbalanced OT
                c = np.vstack(
                    [c, np.append(np.ones(s.shape[0] - 2) * self.alpha * np.max(c[:c.shape[0] - 1, :c.shape[1] - 1]), -1)])

                return c, r, s

        # unicommodity case
        except:

            # sparse ground metric
            c[c >= self.t] = -1

            # Unbalanced OT
            # Swapping histograms: biggest mass -> s (on the "right")
            lambda_g = np.sum(x)
            lambda_h = np.sum(y)
            if lambda_g >= lambda_h:
                s = x
                r = y
                c = c.transpose()  # swapped histograms
            else:
                s = y
                r = x

            # Sparse matrix: add transhipment links
            # extended histogram with transhipment vertex
            r = np.append(r, 0)
            s = np.append(s, 0)

            c = np.vstack([c, np.ones(s.size - 1) * self.t * 0.5])  # cost equal to threshold
            col = np.append(np.ones(r.size - 1) * self.t * 0.5, -1)
            c = np.column_stack((c, col))

            # extended histogram for unbalanced OT
            r = np.append(r, np.sum(s) - np.sum(r))
            s = np.append(s, 0)

            # adding row and col to cost matrix for unbalanced OT
            c = np.vstack(
                [c, np.append(np.ones(s.size - 2) * self.alpha * np.max(c[:c.shape[0] - 1, :c.shape[1] - 1]), -1)])

            return c, r, s

    def index_edges(c, r, s):
        """
        Constructing the edge list of the transport network

        Parameters:
            c, np.array: extended ground distance matrix (accounting for auxiliary nodes)
            r, np.array: extended "right" distribution (accounting for auxiliary nodes)
            s, np.array: extended "left" distribution (accounting for auxiliary nodes)

        Return:
            idx, list: edges list of the transport network
        """

        # original indexes of the bipartite graph
        idx_inner = np.array(list(np.ndindex(c.shape[0] - 2, c.shape[1] - 1)))
        idx_ = np.column_stack(
            (np.zeros(idx_inner.shape)[:, 0], np.ones(idx_inner.shape)[:, 0] * c.shape[0] - 2))
        idx_inner = idx_inner + idx_

        # multicommodity RGB case
        try:
            if r.shape[1] == 3 and s.shape[1] == 3:

                # thresholding cost edges
                extra_node_1 = r.shape[0] + s.shape[0] - 4
                index_list_1_1 = np.array([(extra_node_1, i) for i in range(r.shape[0] - 2)])
                index_list_1_2 = np.array([(i + r.shape[0] - 2, extra_node_1) for i in range(s.shape[0] - 2)])

                # unbalanced OT edges
                extra_node_2 = r.shape[0] + s.shape[0] - 3
                index_list_2 = np.array([(i + r.shape[0] - 2, extra_node_2) for i in range(s.shape[0] - 2)])

                # all indexes
                idx = np.concatenate((idx_inner, index_list_1_1, index_list_1_2, index_list_2), axis=0)

                return idx

        # unicommodity case
        except:

            # thresholding cost edges
            extra_node_1 = r.size + s.size - 4
            index_list_1_1 = np.array([(extra_node_1, i) for i in range(r.size - 2)])
            index_list_1_2 = np.array([(i + r.size - 2, extra_node_1) for i in range(s.size - 2)])

            # unbalanced OT edges
            extra_node_2 = r.size + s.size - 3
            index_list_2 = np.array([(i + r.size - 2, extra_node_2) for i in range(s.size - 2)])

            # all indexes
            idx = np.concatenate((idx_inner, index_list_1_1, index_list_1_2, index_list_2), axis=0)

            return idx

    def cost_edge(c):
        """
        Construct the cost array

        Parameters:
            c, np.array: extended ground distance matrix

        Return:
            length_, np.array: cost of edges
        """

        c_flat_inner = c[:c.shape[0] - 2, :c.shape[1] - 1].flatten()  # cost bipartite graph
        c_flat_1_1 = c[:c.shape[0] - 2, c.shape[1] - 1]  # cost transhipment trimming, inflowing
        c_flat_1_2 = c[c.shape[0] - 2, :c.shape[1] - 1]  # cost transhipment trimming, outflowing
        c_flat_2 = c[c.shape[0] - 1, :c.shape[1] - 1]  # cost penalty unbalanced OT

        length_ = np.concatenate((c_flat_inner, c_flat_1_1, c_flat_1_2, c_flat_2), axis=0)

        return length_

    def topology(e, l):
        """
        Construct the network topology

        Parameters:
            e, list: list of edges in the network (to trim)
            l, np.array: cost of edges

        Return:
            B, sparse matrix: incidence matrix
            length_, np.array: trimmed edges v
        """

        # remove edges over threshold
        length_, edges_ = [], []
        for i in range(l.size):
            if l[i] != -1:
                edges_.append(tuple((int(e[i][0]), int(e[i][1]))))
                length_.append(l[i])
        length_ = np.array(length_)

        nnodes = np.max(edges_) + 1
        g = nx.Graph()
        nodes = np.array(range(nnodes))
        g.add_nodes_from(nodes)
        g.add_edges_from(edges_)
        self.g = g
        B = nx.linalg.graphmatrix.incidence_matrix(g, nodelist=nodes, edgelist=edges_, oriented=True)

        return B, length_

    def rhs_construction(r, s):
        """
        Construct the mass matrix, S

        Parameters:
            r, np.array: extended "right" distribution (accounting for auxiliary nodes)
            s, np.array: extended "left" distribution (accounting for auxiliary nodes)

        Return:
            forcing, np.array: right hand side forcing
        """

        # multicommodity RGB case
        try:
            if r.shape[1] == 3 and s.shape[1] == 3:

                col_0 = (r[:r.shape[0] - 2, 0], -s[:s.shape[0] - 2, 0], 0, r[r.shape[0] - 1, 0])
                col_1 = np.concatenate((r[:r.shape[0] - 2, 1], -s[:s.shape[0] - 2, 1], 0, r[r.shape[0] - 1, 1]), axis=None)
                col_2 = np.concatenate((r[:r.shape[0] - 2, 2], -s[:s.shape[0] - 2, 2], 0, r[r.shape[0] - 1, 2]), axis=None)

                forcing = np.concatenate(col_0, axis=None)
                forcing = np.column_stack((forcing, col_1))
                forcing = np.column_stack((forcing, col_2))

                return forcing

        # unicommodity case
        except:

            forcing = np.concatenate((r[:r.size-2], -s[:s.size-2], 0, r[r.size-1]), axis=None)

            return forcing

    #####################################
    ############# OT SETUP #############
    #####################################

    # topology setup
    if self.verbose: print("* constructing the problem setup")

    self.c, self.r, self.s = sparsunb(self.c, self.x, self.y)
    edges = index_edges(self.c, self.r, self.s)
    length = cost_edge(self.c)
    self.B, self.length = topology(edges, length)
    self.forcing = rhs_construction(self.r, self.s)

    return self.r, self.s, self.c

import igraph as ig
import math
import csv
import numpy as np


class graph_model:
    def __init__(self):
        self._graph = ig.Graph()



    def __create_variable_node(self, v_name, rank=None):
        self._graph.add_vertex(v_name)

    def add_variable_node(self, v_name):
        if self.get_node_status(v_name) != False:
            pass
        else:
            self.__create_variable_node(v_name)

    def get_node_status(self, name):
        if len(self._graph.vs) == 0:
            return False
        elif len(self._graph.vs.select(name_eq=name)) == 0:
            return False
        else:
            return True
    def read_from_csv(self,filename, n_users_tokeep = 100):
        with open(filename, newline='') as csvfile0:
            datareader0 = csv.reader(csvfile0, delimiter=',')
            n_users = np.max(np.array([int(row[0]) for row in datareader0 if row[0] != 'userId'] ))
        with open(filename, newline='') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')
            edge_list = []
            ratings = []
            count = 0
            for row in datareader:
                if row[0] != 'userId':
                    if int(row[0])<n_users_tokeep:
                        self.add_variable_node(str(row[0]))
                        self.add_variable_node(str(2*n_users+int(row[1])))
                        edge_list.append((self._graph.vs.find(name=str(row[0])).index, self._graph.vs.find(name=str(2*n_users+int(row[1]))).index))
                        ratings.append(math.floor(float(row[2])))
        self._graph.add_edges(edge_list)
        self._graph.es['ratings'] = ratings


    def get_graph(self):
        return self._graph
    def is_connected(self):
        return self._graph.is_connected()

    def is_loop(self):
        return any(self._graph.is_loop())
    



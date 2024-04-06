import igraph as ig
import math
import csv
import numpy as np


class graph_model:
    """
    A class representing a probabilistic graphical model.

    Attributes:
        _graph (igraph.Graph): The graph object representing the graphical model.

    Methods:
        __init__(): Initializes the GraphModel with an empty graph.
        __create_variable_node(v_name, rank=None): Creates a variable node in the graph.
        add_variable_node(v_name): Adds a variable node to the graph if it doesn't already exist.
        get_node_status(name): Checks if a node with the given name exists in the graph.
        read_from_csv(filename, n_users_tokeep=100): Reads data from a CSV file and populates the graph.
        get_graph(): Returns the graph object.
        is_connected(): Checks if the graph is connected.
        is_loop(): Checks if the graph contains any loops.
    """
    def __init__(self):
        self._graph = ig.Graph()



    def __create_variable_node(self, v_name, rank=None):
        """
        Creates a variable node in the graph.

        Args:
            v_name (str): Name of the variable node.
            rank (int, optional): Rank of the variable node.
        """
        self._graph.add_vertex(v_name)

    def add_variable_node(self, v_name):
        """
        Adds a variable node to the graph if it doesn't already exist.

        Args:
            v_name (str): Name of the variable node.
        """
        if self.get_node_status(v_name) != False:
            pass
        else:
            self.__create_variable_node(v_name)

    def get_node_status(self, name):
        """
        Checks if a node with the given name exists in the graph.

        Args:
            name (str): Name of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        if len(self._graph.vs) == 0:
            return False
        elif len(self._graph.vs.select(name_eq=name)) == 0:
            return False
        else:
            return True
    def read_from_csv(self,filename, n_users_tokeep = 100):
        """
        Reads data from a CSV file and populates the graph.

        Args:
            filename (str): Path to the CSV file.
            n_users_tokeep (int, optional): Number of users to keep.

        Raises:
            FileNotFoundError: If the specified file is not found.
        """
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
        """
        Returns the graph object.

        Returns:
            igraph.Graph: The graph object representing the graphical model.
        """
        return self._graph
    def is_connected(self):
        return self._graph.is_connected()

    def is_loop(self):
        return any(self._graph.is_loop())
    



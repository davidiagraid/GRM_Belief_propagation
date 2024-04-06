import numpy as np
from graph_models import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score

def sigmoid(x):
    return 1/(1+np.exp(-x))

precision = np.finfo(float).eps

class LBP():
    def __init__(self,pgm):
        if type(pgm) is not graph_model:
            raise Exception('PGM is not a graphical model')
        if not pgm.is_connected():
            raise Exception('PGM is not connected')
        if len(pgm.get_graph().es) - 1 == len(pgm.get_graph().vs):
            raise Exception('PGM is a tree')
        self.msg     = {}
        self.pgm = pgm
        self.belief = {}
        self.potential = {}
        self.seed = []
        self.prop_strength = 0.501
        self.N_obs = []
        self.P_obs = []
        self.AUC_train_list = []
        self.positive_nodes_test = []
        self.negative_nodes_test = []

        # Initialization of messages
        for edge in self.pgm.get_graph().es:
            start_index, end_index = edge.tuple[0], edge.tuple[1]
            start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

            self.msg[(start_name, end_name)] = 0.5
            self.msg[(end_name, start_name)] = self.msg[(start_name, end_name)]

        for v in self.pgm.get_graph().vs:
            self.potential[v['name']] = 0.5
            self.belief[v['name']] = 0.5

    def get_msg(self,name_neigh, v_name):
        return self.msg[(name_neigh, v_name)]

    def compute_belief(self, v_name):
        incoming_messages = []
        for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(v_name)]['name']:
            incoming_messages.append(self.get_msg(name_neighbor, v_name))
        belief_pos = self.potential[v_name]*np.prod(incoming_messages)
        belief_neg =( 1-self.potential[v_name])*np.prod(np.ones(len(incoming_messages))- incoming_messages)
        self.belief[v_name] = belief_pos/(belief_pos+belief_neg)

    def compute_messages(self,v):
        epsilon = self.prop_strength
        start_name = v['name']

        #Message i->j
        for end_name in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(start_name)]['name']:
            msg_pos = epsilon*self.belief[start_name]/self.get_msg(end_name,start_name)+ (1-epsilon)* (1-self.belief[start_name])/(1-self.get_msg(end_name,start_name))
            msg_neg = (1-epsilon)*self.belief[start_name]/self.get_msg(end_name,start_name)+ (epsilon)* (1-self.belief[start_name])/(1-self.get_msg(end_name,start_name))
            self.msg[(start_name,end_name)] = msg_pos/(msg_pos+msg_neg)


    def propagate(self, phi = 0.55):
          for v in self.pgm.get_graph().vs:
              if v in self.P_obs:
                  self.potential[v['name']] = phi
              elif v in self.N_obs:
                  self.potential[v['name']] = 1-phi
              else:
                  self.potential[v['name']] = 0.5

          for v in self.pgm.get_graph().vs:
              self.compute_belief(v['name'])
          for v in self.pgm.get_graph().vs:
              self.compute_messages(v)



    def set_labels(self, k_1 = 100):
        n_vertices = len([v.index for v in self.pgm.get_graph().vs])
        OK = False
        while not OK :
            seed = np.random.choice(n_vertices, size = k_1)
            seed_vertices = self.pgm.get_graph().vs.select(seed) # getting seed vertices and setting their label to 1
            self.pgm.get_graph().vs['label'] = 'undefined' # By default
            seed_vertices['label'] = 1
            for v in seed_vertices:
                incident_edges = self.pgm.get_graph().incident(v.index, mode="all")
                for edge_i in incident_edges:
                    edge = self.pgm.get_graph().es[edge_i]
                    if edge.tuple[1]==v.index:
                        connected_V = edge.tuple[0]
                    else:
                        connected_V = edge.tuple[1]

                    if edge['ratings']==5.0  :
                        self.pgm.get_graph().vs[connected_V]['label'] = 1 #we set positive, all the movies rated 5 by a seed vertex
                    elif  edge['ratings']<5.0  and self.pgm.get_graph().vs[connected_V]['label']== 'undefined':
                        self.pgm.get_graph().vs[connected_V]['label'] = 0
            self.seed = seed
            self.positive_nodes = [v for v in self.pgm.get_graph().vs if v['label']==1]
            self.negative_nodes = [v for v in self.pgm.get_graph().vs if v['label']==0]
            self.labeled_nodes = [v for v in self.pgm.get_graph().vs if v['label']==0 or v['label']==1]

            n_pos = len(self.positive_nodes)
            n_neg = len(self.negative_nodes)
            if min(n_pos, n_neg) >0.75*k_1 and max(n_pos,n_neg)<2*k_1:
                OK = True




    def full_inference(self,obs_rate = 0.05, N_iter = 60):
        obs_index1 = np.random.binomial(1, obs_rate, size = len(self.positive_nodes))
        self.P_obs = [self.positive_nodes[i] for i in range(len(self.positive_nodes)) if obs_index1[i]==1]

        obs_index2 = np.random.binomial(1, obs_rate, size = len(self.negative_nodes))
        self.N_obs = [self.negative_nodes[i] for i in range(len(self.negative_nodes)) if obs_index2[i]==1]

        for a in range(N_iter):
            AUC = self.get_AUC(plot = False)
            self.AUC_train_list.append(AUC)
            self.propagate(phi = 0.55)
            #print(f'AUC train : {AUC_train}')
            #print(f'AUC test : {AUC_test}')


    def get_AUC(self, plot = False):

        train_labeled =  [v for v in self.labeled_nodes]

        labels_train = [v['label'] for v in train_labeled]

        preds_train =  [self.belief[v['name']] for v in train_labeled]
        fpr_train, tpr_train, thresholds_train = roc_curve(labels_train, preds_train)

        if plot :
            display_train = RocCurveDisplay.from_predictions(
            labels_train,preds_train,
            name='Train ROC CURVE',
            color="darkorange")

            _ = display_train.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Train ROC CURVE")
            plt.show()


        AUC = roc_auc_score(labels_train,preds_train,multi_class="ovr",average="micro")


        return AUC
    def plot_AUC_list(self):
        plt.plot(range(len(self.AUC_train_list)), self.AUC_train_list, label = 'AUC train')
        plt.xlabel('Number of iterations')
        plt.ylabel('AUC')
        plt.title('AUC evolution during training')
        plt.legend()
        plt.show()

class SBP():
    def __init__(self,pgm):
        if type(pgm) is not graph_model:
            raise Exception('PGM is not a graphical model')
        if not pgm.is_connected():
            raise Exception('PGM is not connected')
        if len(pgm.get_graph().es) - 1 == len(pgm.get_graph().vs):
            raise Exception('PGM is a tree')
        self.msg     = {}
        self.msg_new = {}
        self.pgm = pgm
        self.msg_prime_base = {}
        self.z = {}
        self.msg_prime = {}
        self.msg_sec = {}
        self.msg_prime_new = {}
        self.belief = {}
        self.b_prime = {}
        self.potential = {}
        self.weight = 0.001*np.ones(6)

        self.N_obs = []
        self.P_obs = []
        self.P_trn = []
        self.N_trn = []
        self.weight_list= []
        self.AUC_train_list = []
        self.AUC_test_list = []
        self.belief_without_1 = {}


        # Initialization of messages
        for edge in self.pgm.get_graph().es:
            start_index, end_index = edge.tuple[0], edge.tuple[1]
            start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

            self.msg[(start_name, end_name)] = 0.5
            self.msg[(end_name, start_name)] = self.msg[(start_name, end_name)]

            self.msg_new[(start_name, end_name)] = 0
            self.msg_new[(end_name, start_name)] = 0

            self.msg_prime_base[(start_name, end_name)] = 0
            self.msg_prime_base[(end_name, start_name)] = 0

            self.z[(start_name, end_name)] = 0
            self.z[(end_name, start_name)] = 0

            self.msg_prime[(start_name, end_name)] = 0
            self.msg_prime[(end_name, start_name)] = 0

            self.msg_sec[(start_name, end_name)] = 0
            self.msg_sec[(end_name, start_name)] = 0

            self.msg_prime_new[(start_name, end_name)] = 0
            self.msg_prime_new[(end_name, start_name)] = 0


        for v in self.pgm.get_graph().vs:
            self.potential[v['name']] = 0.5
            self.belief[v['name']] = 0.5

    def get_msg(self,name_neigh, v_name):
        return self.msg[(name_neigh, v_name)]

    def norm_small(self,pos,neg):#Normalization for small numbers
        return np.exp(np.log(pos)-np.log(pos+ neg))

    def compute_belief(self, v_name):
        incoming_messages = []
        for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(v_name)]['name']:
            incoming_messages.append(self.get_msg(name_neighbor, v_name))
        belief_pos = self.potential[v_name]*np.prod(incoming_messages)
        belief_neg =( 1-self.potential[v_name])*np.prod(np.ones(len(incoming_messages))- incoming_messages)

        #for neigh in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(v_name)]['name']:# Storing beliefs without 1 message
        #    incoming_without_neigh = np.prod([self.get_msg(name_neighbor, v_name)] for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(v_name)]['name'] if name_neighbor != neigh)

        self.belief[v_name] = self.norm_small(belief_pos, belief_neg)
        #self.belief[v_name] = belief_pos/(belief_pos+belief_neg)

    def compute_message(self,edge):
        start_index, end_index = edge.tuple[0], edge.tuple[1]
        start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

        epsilon = self.compute_epsilon(edge)

        #Message i->j
        incoming_messages_i = []
        for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(start_name)]['name']:
            if name_neighbor != end_name:
                incoming_messages_i.append(self.get_msg(name_neighbor, start_name))
        nb_inc = len(incoming_messages_i)
        incoming_messages_i = np.array(incoming_messages_i)

        msg_pos = epsilon*self.potential[start_name]* np.prod(incoming_messages_i)+ (1-epsilon)* (1-self.potential[start_name])*np.prod(np.ones(nb_inc)- incoming_messages_i)
        msg_neg = (1-epsilon)*self.potential[start_name]* np.prod(incoming_messages_i)+ (epsilon)* (1-self.potential[start_name])*np.prod(np.ones(nb_inc)- incoming_messages_i)

        self.msg[(start_name, end_name)] = self.norm_small(msg_pos, msg_neg)
        #else:
        #    self.msg[(start_name, end_name)] = msg_pos/(msg_pos+msg_neg)

        #Message j->i
        incoming_messages_j = []
        for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(end_name)]['name']:
            if name_neighbor != start_name:
                incoming_messages_j.append(self.get_msg(name_neighbor, end_name))
        nb_inc = len(incoming_messages_j)
        incoming_messages_j = np.array(incoming_messages_j)

        msg_pos = epsilon*self.potential[end_name]* np.prod(incoming_messages_j)+ (1-epsilon)* (1-self.potential[end_name])*np.prod(np.ones(nb_inc)- incoming_messages_j)
        msg_neg = (1-epsilon)*self.potential[end_name]* np.prod(incoming_messages_j)+ (epsilon)* (1-self.potential[end_name])*np.prod(np.ones(nb_inc)- incoming_messages_j)

        self.msg[(end_name, start_name)] = self.norm_small(msg_pos,msg_neg)
        #else:
        #    self.msg[(end_name, start_name)] = msg_pos/(msg_pos+msg_neg)

    def compute_epsilon(self, edge):
        theta = np.zeros(6)
        rating = float(edge['ratings'])
        theta[math.floor(rating)] = 1
        epsilon = sigmoid(theta.T.dot(self.weight))
        return epsilon
    def compute_theta(self,edge):
        theta = np.zeros(6)
        rating = float(edge['ratings'])
        theta[math.floor(rating)] = 1
        return theta


    def differentiate(self, eta = 1):
        for edge in self.pgm.get_graph().es:
            start_index, end_index = edge.tuple[0], edge.tuple[1]
            start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

            epsilon = self.compute_epsilon(edge)
            theta = self.compute_theta(edge)

            b_i = self.belief[start_name]
            b_j = self.belief[end_name]
            m_ji = self.msg[(end_name, start_name)]
            m_ij = self.msg[(start_name, end_name)]

            self.msg_prime_base[(start_name, end_name)] = (epsilon*(1-epsilon)*(b_i-m_ji)*theta)/(b_i+ m_ji-2*b_i*m_ji)
            self.z[(start_name, end_name)] = (epsilon-m_ji-m_ij+2*m_ij*m_ji)/(b_i+m_ji-2*b_i*m_ji)

            self.msg_prime_base[(end_name, start_name)] = (epsilon*(1-epsilon)*(b_j-m_ij)*theta)/(b_j+ m_ij-2*b_j*m_ij)
            self.z[(end_name, start_name)] = (epsilon-m_ij-m_ji+2*m_ij*m_ji)/(b_j+m_ij-2*b_j*m_ij)

        for edge in self.pgm.get_graph().es:
            start_index, end_index = edge.tuple[0], edge.tuple[1]
            start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

            self.msg_prime[(start_name, end_name)] = self.msg_prime_base[(start_name, end_name)]
            self.msg_prime[(end_name, start_name)] = self.msg_prime_base[(end_name, start_name)]
        for a in range(eta):
            for edge in self.pgm.get_graph().es:
                start_index, end_index = edge.tuple[0], edge.tuple[1]
                start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

                #b_j = self.belief[end_name]
                #b_i = self.belief[start_name]
                #m_ij = self.msg[(start_name, end_name)],precision
                #m_ji = self.msg[(end_name, start_name)], precision

                #self.msg_sec[(start_name, end_name)]  = (b_j*(1-b_j)*self.msg_prime[(start_name, end_name)])/(m_ij*(1-m_ij))
                #self.msg_sec[(end_name, start_name)]  = (b_i*(1-b_i)*self.msg_prime[(end_name, start_name)])/(m_ji*(1-m_ji))

                incoming_msgs_j_red = []
                incoming_msgs_j = []
                for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(end_name)]['name']:

                    incoming_msgs_j.append(self.get_msg(name_neighbor, end_name))
                    if name_neighbor != start_name:
                        incoming_msgs_j_red.append(self.get_msg(name_neighbor, end_name))

                belief_pos_j_red = self.potential[end_name]*np.prod(incoming_msgs_j_red)
                belief_pos_j = self.potential[end_name]*np.prod(incoming_msgs_j)
                belief_neg_j_red =( 1-self.potential[end_name])*np.prod(np.ones(len(incoming_msgs_j_red))- incoming_msgs_j_red)
                belief_neg_j =( 1-self.potential[end_name])*np.prod(np.ones(len(incoming_msgs_j))- incoming_msgs_j)

                b_j_m_ij = np.exp(np.log(belief_pos_j_red) - np.log(belief_pos_j + belief_neg_j) )
                b_jO_m_ijO = np.exp(np.log(belief_neg_j_red) - np.log(belief_pos_j + belief_neg_j) )



                self.msg_sec[(start_name, end_name)]  = b_j_m_ij*b_jO_m_ijO * self.msg_prime[(start_name, end_name)]



                incoming_msgs_i_red = []
                incoming_msgs_i = []
                for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(start_name)]['name']:

                    incoming_msgs_i.append(self.get_msg(name_neighbor, start_name))
                    if name_neighbor != start_name:
                        incoming_msgs_i_red.append(self.get_msg(name_neighbor, start_name))

                belief_pos_i_red = self.potential[start_name]*np.prod(incoming_msgs_i_red)
                belief_pos_i = self.potential[start_name]*np.prod(incoming_msgs_i)
                belief_neg_i_red =( 1-self.potential[start_name])*np.prod(np.ones(len(incoming_msgs_i_red))- incoming_msgs_i_red)
                belief_neg_i =( 1-self.potential[start_name])*np.prod(np.ones(len(incoming_msgs_i))- incoming_msgs_i)

                b_i_m_ji = np.exp(np.log(belief_pos_i_red) - np.log(belief_pos_i + belief_neg_i) )
                b_iO_m_jiO = np.exp(np.log(belief_neg_i_red) - np.log(belief_pos_i + belief_neg_i) )



                self.msg_sec[(end_name, start_name)]  = b_i_m_ji*b_iO_m_jiO * self.msg_prime[(end_name, start_name)]


            for edge in self.pgm.get_graph().es:
                start_index, end_index = edge.tuple[0], edge.tuple[1]
                start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

                # i->j
                res = np.zeros(6)
                for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(start_name)]['name']:
                    if name_neighbor != end_name:
                        res+=  self.msg_sec[(name_neighbor, start_name)]
                res*= self.z[(start_name, end_name)]

                self.msg_prime_new[(start_name, end_name)] = self.msg_prime_base[(start_name, end_name)] + res

                #j->i
                res = np.zeros(6)
                for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(end_name)]['name']:
                    if name_neighbor != start_name:
                        res+=  self.msg_sec[(name_neighbor, end_name)]
                res*= self.z[(end_name, start_name)]

                self.msg_prime_new[(end_name, start_name)] = self.msg_prime_base[(end_name, start_name)] + res

            for edge in self.pgm.get_graph().es:
                start_index, end_index = edge.tuple[0], edge.tuple[1]
                start_name, end_name = self.pgm.get_graph().vs[start_index]['name'], self.pgm.get_graph().vs[end_index]['name']

                self.msg_prime[(start_name, end_name)] = self.msg_prime_new[(start_name, end_name)]
                self.msg_prime[(end_name, start_name)] = self.msg_prime_new[(end_name, start_name)]

        for v in self.pgm.get_graph().vs:
            v_name = v['name']
            b_j = self.belief[v_name]

            res = np.zeros(6)
            for name_neighbor in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(v_name)]['name']:
                #m_ij = self.msg[(name_neighbor, v_name)], precision

                incoming_msgs_j_red = []
                incoming_msgs_j = []
                for name_neighbor2 in self.pgm.get_graph().vs[self.pgm.get_graph().neighbors(v_name)]['name']:

                    incoming_msgs_j.append(self.get_msg(name_neighbor2, v_name))
                    if name_neighbor2 != name_neighbor:
                        incoming_msgs_j_red.append(self.get_msg(name_neighbor2, v_name))

                belief_pos_j_red = self.potential[v_name]*np.prod(incoming_msgs_j_red)
                belief_pos_j = self.potential[v_name]*np.prod(incoming_msgs_j)
                belief_neg_j_red =( 1-self.potential[v_name])*np.prod(np.ones(len(incoming_msgs_j_red))- incoming_msgs_j_red)
                belief_neg_j =( 1-self.potential[v_name])*np.prod(np.ones(len(incoming_msgs_j))- incoming_msgs_j)

                b_j_m_ij = np.exp(np.log(belief_pos_j_red) - np.log(belief_pos_j + belief_neg_j) )
                b_jO_m_ijO = np.exp(np.log(belief_neg_j_red) - np.log(belief_pos_j + belief_neg_j) )


                #res+=  self.msg_prime[(name_neighbor, v_name)]* b_j*(1-b_j)/(m_ij*(1-m_ij))
                res+=  self.msg_prime[(name_neighbor, v_name)]*b_jO_m_ijO*b_j_m_ij

            self.b_prime[v_name] = res




    def propagate(self, N_iter_msg = 5, phi = 0.55, entire = False):
        if not entire:
            for v in self.pgm.get_graph().vs:
                if v in self.P_obs:
                    self.potential[v['name']] = phi
                elif v in self.N_obs:
                    self.potential[v['name']] = 1-phi
                else:
                    self.potential[v['name']] = 0.5

            for n in range(N_iter_msg):
                for v in self.pgm.get_graph().vs:
                    self.compute_belief(v['name'])
                for edge in self.pgm.get_graph().es:
                    self.compute_message(edge)
        else:
            for v in self.pgm.get_graph().vs:
                if v in self.positive_nodes:
                    self.potential[v['name']] = phi
                elif v in self.negative_nodes:
                    self.potential[v['name']] = 1-phi
                else:
                    self.potential[v['name']] = 0.5

            for n in range(N_iter_msg):
                for v in self.pgm.get_graph().vs:
                    self.compute_belief(v['name'])
                for edge in self.pgm.get_graph().es:
                    self.compute_message(edge)

    def weight_update(self, lambd = 0.05, d = 0.0001, alpha = 0.00001, beta = 0.00001):
        self.differentiate()
        w_prime = 2* lambd*self.weight
        for p in self.P_trn:
            for n in self.N_trn:
                h = sigmoid((self.belief[n['name']]- self.belief[p['name']])/d)
                w_prime +=  (h*(1-h)* (self.b_prime[n['name']]- self.b_prime[p['name']]))/d

        self.weight = self.weight - max(alpha, beta/(np.linalg.norm(w_prime)))* w_prime

    def set_labels(self, k_1 = 100):
        n_vertices = len([v.index for v in self.pgm.get_graph().vs])
        OK = False
        while not OK :
            seed_vertices = self.pgm.get_graph().vs.select(np.random.choice(n_vertices, size = k_1)) # getting seed vertices and setting their label to 1
            self.pgm.get_graph().vs['label'] = 'undefined' # By default
            seed_vertices['label'] = 1
            for v in seed_vertices:
                incident_edges = self.pgm.get_graph().incident(v.index, mode="all")
                for edge_i in incident_edges:
                    edge = self.pgm.get_graph().es[edge_i]
                    if edge.tuple[1]==v.index:
                        connected_V = edge.tuple[0]
                    else:
                        connected_V = edge.tuple[1]

                    if edge['ratings']==5.0  :
                        self.pgm.get_graph().vs[connected_V]['label'] = 1 #we set positive, all the movies rated 5 by a seed vertex
                    elif  edge['ratings']<5.0  and self.pgm.get_graph().vs[connected_V]['label']== 'undefined':
                        self.pgm.get_graph().vs[connected_V]['label'] = 0

            self.positive_nodes = [v for v in self.pgm.get_graph().vs if v['label']==1]
            self.negative_nodes = [v for v in self.pgm.get_graph().vs if v['label']==0]
            self.labeled_nodes = [v for v in self.pgm.get_graph().vs if v['label']==0 or v['label']==1]

            n_pos = len(self.positive_nodes)
            n_neg = len(self.negative_nodes)
            if min(n_pos, n_neg) >0.75*k_1 and max(n_pos,n_neg)<2*k_1:
                OK = True



    def full_inference(self,obs_rate = 0.3, N_iter = 60, N_iter_msg = 5, lambd = 0.05, d = 0.0001, alpha = 0.000001, beta = 0.000001):
        n_vertices = len([v.index for v in self.pgm.get_graph().vs])

        obs_index1 = np.random.binomial(1, obs_rate, size = len(self.positive_nodes))
        self.P_obs = [self.positive_nodes[i] for i in range(len(self.positive_nodes)) if obs_index1[i]==1]
        self.P_trn = [v for v in self.positive_nodes if v not in self.P_obs ]

        obs_index2 = np.random.binomial(1, obs_rate, size = len(self.negative_nodes))
        self.N_obs = [self.negative_nodes[i] for i in range(len(self.negative_nodes)) if obs_index2[i]==1]
        self.N_trn = [v for v in self.negative_nodes if v not in self.N_obs ]
        self.weight_list= []
        self.weight_list.append(0.001*np.ones(6))
        for a in range(N_iter):
            self.propagate( N_iter_msg, phi = 0.55)
            self.weight_update(lambd = 0.05, d = 0.0001, alpha = 0.00001, beta = 0.00001)
            self.weight_list.append(self.weight)
            AUC_train, AUC_test = self.get_AUC(plot = False)
            self.AUC_train_list.append(AUC_train)
            self.AUC_test_list.append(AUC_test)
            #print(f'AUC train : {AUC_train}')
            #print(f'AUC test : {AUC_test}')

        self.propagate(phi = 0.55,N_iter_msg = 20, entire = True)


    def plot_weight_evolution(self, N_iterations):
        weights = np.array(self.weight_list)

        for i in range(6):
            plt.plot(range(N_iterations+1), weights[:N_iterations+1,i] , label = f'w_{i}')
        plt.xlabel('Number of iterations')
        plt.ylabel('Weight coefficients')
        plt.legend()
        plt.show()

    def get_AUC(self, plot = False):

        train_labeled =  [v for v in self.labeled_nodes if v in self.P_trn or v in self.N_trn]
        test_labeled = [v for v in self.labeled_nodes if v in self.P_obs or v in self.N_obs]

        labels_train = [v['label'] for v in train_labeled]
        labels_test = [v['label'] for v in test_labeled]

        preds_train =  [self.belief[v['name']] for v in train_labeled]
        preds_test =  [self.belief[v['name']] for v in test_labeled]

        fpr_train, tpr_train, thresholds_train = roc_curve(labels_train, preds_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(labels_test, preds_test)

        if plot :
            display_train = RocCurveDisplay.from_predictions(
            labels_train,preds_train,
            name='Train ROC CURVE',
            color="darkorange")

            _ = display_train.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Train ROC CURVE")
            plt.show()

            display_test = RocCurveDisplay.from_predictions(
            labels_test,
            preds_test,
            name='Test ROC CURVE',
            color="darkorange")
            _ = display_test.ax_.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Test ROC CURVE")
            plt.show()

        AUC_train_micro = roc_auc_score(labels_train,preds_train,multi_class="ovr",average="micro")
        AUC_test_micro = roc_auc_score(labels_test,preds_test,multi_class="ovr",average="micro")


        return AUC_train_micro, AUC_test_micro
    def plot_AUC_list(self):
        plt.plot(range(len(self.AUC_train_list)), self.AUC_train_list, label = 'AUC training')
        plt.xlabel('Number of iterations')
        plt.ylabel('AUC')
        plt.title('AUC evolution during training')
        plt.legend()
        plt.show()

    def copy_labels(self, seed_list,k_1=100):
        n_vertices = len([v.index for v in self.pgm.get_graph().vs])
        for i in range(1) :
            seed_vertices = self.pgm.get_graph().vs.select(seed_list) # getting seed vertices and setting their label to 1
            self.pgm.get_graph().vs['label'] = 'undefined' # By default
            seed_vertices['label'] = 1
            for v in seed_vertices:
                incident_edges = self.pgm.get_graph().incident(v.index, mode="all")
                for edge_i in incident_edges:
                    edge = self.pgm.get_graph().es[edge_i]
                    if edge.tuple[1]==v.index:
                        connected_V = edge.tuple[0]
                    else:
                        connected_V = edge.tuple[1]

                    if edge['ratings']==5.0  :
                        self.pgm.get_graph().vs[connected_V]['label'] = 1 #we set positive, all the movies rated 5 by a seed vertex
                    elif  edge['ratings']<5.0  and self.pgm.get_graph().vs[connected_V]['label']== 'undefined':
                        self.pgm.get_graph().vs[connected_V]['label'] = 0

            self.positive_nodes = [v for v in self.pgm.get_graph().vs if v['label']==1]
            self.negative_nodes = [v for v in self.pgm.get_graph().vs if v['label']==0]
            self.labeled_nodes = [v for v in self.pgm.get_graph().vs if v['label']==0 or v['label']==1]

            n_pos = len(self.positive_nodes)
            n_neg = len(self.negative_nodes)

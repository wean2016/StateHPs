import argparse
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import pickle
import os
from itertools import product
from scipy import integrate
from scipy.special import digamma
from scipy.special import logsumexp
from typing import Dict, List, Tuple, Any

__MIN__ = -100000000000000

class TimeSlot(object):
    def __init__(self, start_time, end_time, in_Cluster_id, event_names, decay, Gao_para, beta):
        self.start_time = start_time
        self.end_time = end_time
        self.T_n = self.end_time - self.start_time
        self.event_names = event_names
        self.event_count = 0
        self.events = []
        self.beta_h = beta
        self.penalty = 'BIC'
        self.decay = decay
        self.Gao_para = Gao_para
        self.Gao_mu_values = Gao_para[0]
        self.Gao_sigma_values = Gao_para[1]
        self.in_Cluster = in_Cluster_id
        self.kernel_sum = None
        self.integral_1 = None
        self.integral_2 = None
        self.event_record_by_event_type_dict = None
        self.last_occur_timestamp = None
        self.current_occur_timestamp = None
        self.v = len(self.event_names)
        self.init_structure = np.eye(self.v, self.v)
    def init_topology(self):
        self.topo_mat = np.zeros([1, 1])
        self.topo = nx.from_numpy_array(self.topo_mat)
        self.node_list = np.array(self.topo.nodes)
        self.A = nx.to_numpy_array(self.topo).astype(float)
        D = np.diag(self.A.sum(axis=0))
        D_inv = np.diag(1.0 / np.where(np.diag(D) == 0, np.inf, np.diag(D)))
        D_inv[np.isinf(D_inv)] = 0
        self.A_sys = D_inv @ self.A @ D_inv
        self.A_k = np.zeros((1, self.v, self.v))
        self.A_sys_k = np.zeros((1, self.v, self.v))
        self.A_k[0] = np.eye(self.v)
        self.A_sys_k[0] = np.eye(self.v)
    def cal_cache(self):
        self.events_dataframe = pd.DataFrame(self.events).reset_index()
        self.max_t = self.events_dataframe['Start Time Stamp'].max()
        self.min_t = self.events_dataframe['Start Time Stamp'].min()
        self.T_n = (self.max_t-self.min_t)*len(self.events_dataframe['Node'].unique())
        self.events_dataframe['Event Ind'] = self.events_dataframe['Event Ind'].to_numpy().astype(np.int64)
        self.events_dataframe['Event Name'] = self.events_dataframe['Event Name'].to_numpy().astype(np.int64)
        self.hist_likelihood = dict()
        for i in range(len(self.event_names)):
            self.hist_likelihood[i] = dict()

        self.max_t = self.events_dataframe['Start Time Stamp'].max()
        self.min_t = self.events_dataframe['Start Time Stamp'].min()
        self.T = (self.max_t - self.min_t) * len(self.events_dataframe['Node'].unique())
        self.event_table_groupby_NE = self.events_dataframe.groupby('Node')
        self.init_structure()
        self.effect_tensor_decay_all = self.get_decay_effect_of_each_jump()
        self.decay_effect_without_end_time = self.get_decay_effect_integral_on_t()

        event_record_num = self.events_dataframe.shape[0]
        time_diff_matrix = np.zeros((event_record_num, event_record_num))

        time_array = self.events_dataframe['Start Time Stamp'].to_numpy()
        for i in range(event_record_num):
            time_diff_matrix[i, :] = time_array[i] - time_array

        self.kernel_sum = np.exp(-self.decay * time_diff_matrix)

        current_occur_timestamp = self.events_dataframe['Start Time Stamp'].to_numpy()
        last_occur_timestamp = np.zeros(event_record_num)
        event_record_by_event_type_dict = {even_type: [] for even_type in self.event_names}

        for event_name in self.event_names:
            event_record_by_event_type_dict[event_name] = self.events_dataframe[self.events_dataframe['Event Name'] == event_name].index.to_numpy()
            last_timestamp = np.zeros_like(event_record_by_event_type_dict[event_name])
            last_timestamp[0] = self.start_time
            last_timestamp[1:] = self.events_dataframe.iloc[event_record_by_event_type_dict[event_name][:-1]]['Start Time Stamp'].to_numpy()
            last_occur_timestamp[event_record_by_event_type_dict[event_name]] = last_timestamp

        self.event_record_by_event_type_dict = event_record_by_event_type_dict
        self.last_occur_timestamp = last_occur_timestamp
        self.current_occur_timestamp = current_occur_timestamp

    def K_hop_neibors(self,node,K):
        if K==0:
            return {node}
        else:
            return (set(nx.single_source_dijkstra_path_length(self.topo, node, K).keys()) - set(
            nx.single_source_dijkstra_path_length(self.topo, node, K-1).keys()))

    def get_decay_effect_of_each_jump(self):
        effect_tensor_decay_all = np.zeros([1, len(self.events_dataframe), len(self.event_names)])
        for k in range(1):
            event_table_array = self.events_dataframe[['Node', 'Start Time Stamp', 'Event Ind', 'Times']].values
            j = 0
            pre_effect = np.zeros(self.v)
            for item_ind in tqdm(range(len(self.events_dataframe))):
                node, start_t, ala_i, times = event_table_array[item_ind, [0, 1, 2, 3]]
                last_node, last_start_t, last_ala_i, last_times = event_table_array[
                    item_ind - 1, [0, 1, 2, 3]]
                if ((last_node != node) or (last_start_t > start_t)):
                    j = 0
                    pre_effect = np.zeros(self.v)
                    try:
                        K_hop_neighbors_NE = self.K_hop_neibors(node, k)
                        neighbors_table = pd.concat(
                            [self.event_table_groupby_NE.get_group(i) for i in K_hop_neighbors_NE])
                        neighbors_table = neighbors_table.sort_values('Start Time Stamp')
                        neighbors_table_value = neighbors_table[
                            ['Node', 'Start Time Stamp', 'Event Ind', 'Times']].values
                    except ValueError as e:
                        K_hop_neighbors_NE = []
                if (len(K_hop_neighbors_NE) == 0):
                    continue
                cur_effect = pre_effect * np.exp((np.min((last_start_t - start_t, 0))) * self.decay)
                while (1):
                    try:
                        nei_node, nei_start_t, nei_ala_i, nei_times = neighbors_table_value[j, :]
                    except Exception as e:
                        break
                    if (nei_start_t < start_t):
                        cur_effect[int(nei_ala_i)] += nei_times * np.exp((nei_start_t - start_t) * self.decay)*self.A_sys_k[k,self.get_node_ind(nei_node),self.get_node_ind(node)]
                        j += 1
                    else:
                        break
                pre_effect = cur_effect
                effect_tensor_decay_all[k, item_ind] = pre_effect
        return effect_tensor_decay_all


    def get_decay_effect_integral_on_t(self):
        decay_effect_without_end_time = np.zeros([len(self.event_names), 1])
        for k in tqdm(range(1)):
            decay_effect_without_end_time[:, k] = self.events_dataframe.groupby('Event Ind').apply(lambda i: (
                    (((1 - np.exp(-self.decay * (self.max_t - i['Start Time Stamp']))) / self.decay) * i['Times']) *
                    i['Node'].apply(lambda j: self.A_sys_k[k, self.get_node_ind(j), :].sum())).sum())
        return decay_effect_without_end_time

    def rou_nk_Q_lower_bound(self,cluster):

        alpha = np.ones([1, len(self.event_names), len(self.event_names)])
        alpha[0] = cluster.alpha_k
        alpha = alpha * cluster.beta_h
        mu = cluster.mu_k
        L = 0

        for i in range(len(self.event_names)):
            ind = np.where(self.events_dataframe['Event Ind'] == i)
            X_i = self.events_dataframe['Times'].values[ind]
            X_i_all = np.zeros_like(self.events_dataframe['Times'].values)
            X_i_all[ind] = X_i
            lambda_i_sum = (self.decay_effect_without_end_time * alpha[:, :, i].T).sum() + mu[i] * self.T
            lambda_for_i = np.zeros(len(self.events_dataframe)) + mu[i]
            for k in range(1):
                lambda_for_i += np.matmul(self.effect_tensor_decay_all[k, :], alpha[k, :, i].T)
            lambda_for_i = lambda_for_i[ind]
            X_log_lambda = (X_i * np.log(lambda_for_i)).sum()
            L += -lambda_i_sum + X_log_lambda
            i += 1
        return L

    def EM_para_for_cluster(self,edge_mat,cluster):

        alpha = np.ones([1, len(self.event_names), len(self.event_names)])
        alpha[0] = cluster.alpha_k
        alpha = alpha * edge_mat
        mu = cluster.mu_k
        L = 0
        mu_q_ii_k=np.zeros(len(self.event_names))
        alpha_q_ijm_k=np.zeros([1, len(self.event_names), len(self.event_names)], dtype=object)
        for i in range(len(self.event_names)):
            Pa_i = set(np.where(edge_mat[:, i] == 1)[0])
            ind = np.where(self.events_dataframe['Event Ind'] == i)
            X_i = self.events_dataframe['Times'].values[ind]
            X_i_all = np.zeros_like(self.events_dataframe['Times'].values)
            X_i_all[ind] = X_i

            lambda_i_sum = (self.decay_effect_without_end_time * alpha[:, :, i].T).sum() + mu[i] * self.T

            lambda_for_i = np.zeros(len(self.events_dataframe)) + mu[i]
            for k in range(1):
                lambda_for_i += np.matmul(self.effect_tensor_decay_all[k, :], alpha[k, :, i].T)
            lambda_for_i = lambda_for_i[ind]
            epsilon = np.finfo(float).eps
            lambda_for_i_safe = np.maximum(lambda_for_i, epsilon)
            X_log_lambda = (X_i * np.log(lambda_for_i_safe)).sum()

            new_Li = -lambda_i_sum + X_log_lambda
            Li = new_Li
            L += Li
            # update mu
            if(mu[i]==0):
                mu_q_ii_k[i] = 0
            else:
                mu_q_ii_k[i] = ((mu[i] / lambda_for_i) * X_i).sum()

            # update alpha
            for j in Pa_i:
                for k in range(1):
                    upper = ((alpha[k, j, i] * ((self.effect_tensor_decay_all)[k, :, j])[
                        ind] / lambda_for_i) * (X_i)).sum()
                    lower = self.decay_effect_without_end_time[j, k]

                    if np.isnan(upper) or np.isnan(lower):
                        alpha_q_ijm_k[0, j, i] = (0,0)
                    else:
                        alpha_q_ijm_k[0, j, i] = (upper,lower)

        return L - (len(self.event_names) +  (edge_mat).sum() * 1) * np.log(
            self.events_dataframe['Times'].sum()) / 2, alpha, mu,mu_q_ii_k,alpha_q_ijm_k

class Cluster(object):
    def __init__(self,  event_num, id, alpha_cluster, event_names, decay, Gao_para, beta_h):
        self.q_ijm_k = None
        self.q_ii_k = None
        self.id = id
        self.event_num = event_num
        self.time_slots = []
        self.sigma_exp = np.full((self.event_num, self.event_num), 0.01)
        self.beta_Rali = np.full(self.event_num, 0.008)
        self.event_names = event_names
        self.decay = decay
        self.Gao_para = Gao_para
        self.Gao_mu_values = Gao_para[0]
        self.Gao_sigma_values = Gao_para[1]
        self.beta_h = beta_h
        self.random_seed = 4
        self.alpha_k = np.full((self.event_num, self.event_num), 0.01)
        self.mu_k = np.full(self.event_num, 0.0008)
        self.alpha_0 = alpha_cluster
        self.pi = 0

    def calculate_all_i(self,time_slot:TimeSlot):
        all_i_sum = 0
        record_event_type = time_slot.events_dataframe['Event Ind'].to_numpy()
        sigma_k_vi_vj = self.sigma_exp.T[np.ix_(record_event_type,record_event_type)]
        beta_h_vi_vj =  self.beta_h.T[np.ix_(record_event_type,record_event_type)]
        inner_kernel_sum = sigma_k_vi_vj * time_slot.kernel_sum
        first_term = beta_h_vi_vj * inner_kernel_sum
        second_term_numerator = np.tril((beta_h_vi_vj ** 2) * (inner_kernel_sum ** 2), k=-1)
        second_term_denominator = np.tril(beta_h_vi_vj * inner_kernel_sum, k=-1)
        for event_name in self.event_names:
            total_sum_first = 0
            total_sum_second = 0
            total_sum_third = 0
            current_first_term_sum = np.sum(np.tril(first_term, k=-1)[time_slot.event_record_by_event_type_dict[event_name]], axis=1)
            total_sum_first += np.sum(np.log(np.sqrt(np.pi / 2) * self.beta_Rali[event_name] + current_first_term_sum))
            current_second_term_numerator = np.sum(np.tril(second_term_numerator, k=-1)[time_slot.event_record_by_event_type_dict[event_name]], axis=1)
            numerator = (2 - np.pi / 2) * self.beta_Rali[event_name] ** 2 + current_second_term_numerator
            current_second_term_denominator = np.sum(np.tril(second_term_denominator, k=-1)[time_slot.event_record_by_event_type_dict[event_name]], axis=1)
            denominator = 2 * (np.sqrt(np.pi / 2) * self.beta_Rali[event_name] + current_second_term_denominator) ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(numerator, denominator)
                result = np.where(np.isnan(result), 0, result)  # 将 NaN 替换为 0
                total_sum_second += np.sum(result)
            current_total_sum_third_integral = (time_slot.decay_effect_without_end_time * self.sigma_exp[:, event_name].T).sum()
            total_sum_third += np.sqrt(np.pi / 2) * self.beta_Rali[event_name] * time_slot.T_n+ current_total_sum_third_integral
            all_i_sum += total_sum_first - total_sum_second - total_sum_third
            if np.isnan(all_i_sum):
                print("q")
        return all_i_sum


class Simulation_NSNID_HP(object):

    def __init__(self, event_table: pd.DataFrame, decay, Gau_para, beta_h_0,alpha_range,parameter_ranges,penalty='BIC',
                  n_hawkes=2, kernel_m=4,time_slot_length=100):

        if (penalty not in {'BIC', 'AIC'}):
            raise Exception('Penalty is not supported')
        self.penalty = penalty
        self.Gau_para = Gau_para
        self.parameter_ranges = parameter_ranges

        self.event_table, self.event_names = self.get_event_table(event_table)
        self.v = len(self.event_names)
        self.alpha_range = alpha_range
        self.decay = decay
        self.kernel_m = kernel_m

        self.max_t = self.event_table['Start Time Stamp'].max()
        self.min_t = self.event_table['Start Time Stamp'].min()
        self.T = (self.max_t - self.min_t) * len(self.event_table['Node'].unique())
        self.init_topology()

        self.beta = beta_h_0
        self.n_clusters = n_hawkes
        self.alpha_clusters = [5] * self.n_clusters
        self.cluster_list = []
        self.init_clusters()
        self.cluster_pi_list = np.random.dirichlet(self.alpha_clusters, size=1).flatten()
        for idx, cluster in enumerate(self.cluster_list):
            cluster.pi = self.cluster_pi_list[idx]

        self.time_slot_length = time_slot_length
        self.num_time_slots = int(self.T / self.time_slot_length)
        self.Z = self.initialize_Z(self.num_time_slots, self.n_clusters)
        self.time_slots = self.generate_time_slots(self.time_slot_length, self.Z, self.beta)
        self.last_matrix = np.zeros((len(self.time_slots), len(self.cluster_list)))

        self.cal_cache()

    def init_topology(self):
        self.topo_mat = np.zeros([1, 1])
        self.topo = nx.from_numpy_array(self.topo_mat)
        self.node_list = np.array(self.topo.nodes)

        self.A = nx.to_numpy_array(self.topo).astype(float)
        D = np.diag(self.A.sum(axis=0))
        D_inv = np.diag(1.0 / np.where(np.diag(D) == 0, np.inf, np.diag(D)))
        D_inv[np.isinf(D_inv)] = 0
        self.A_sys = D_inv @ self.A @ D_inv

        self.A_k = np.zeros((1, self.v, self.v))
        self.A_sys_k = np.zeros((1, self.v, self.v))
        self.A_k[0] = np.eye(self.v)
        self.A_sys_k[0] = np.eye(self.v)

    def init_clusters(self):
        """
        Dynamically initialize each cluster's parameters by sampling
        from its own alpha/mu range, so that clusters are not identically initialized.
        """
        self.cluster_list = []
        for i in range(self.n_clusters):
            key = f"hawkes_{i + 1}"
            alpha_min, alpha_max = self.parameter_ranges[key]['alpha_range']
            mu_min, mu_max = self.parameter_ranges[key]['mu_range']

            alpha_init = np.random.uniform(alpha_min, alpha_max, size=(self.v, self.v))
            sigma_init = alpha_init.copy()

            mu_init = np.random.uniform(mu_min, mu_max, size=(self.v,))
            beta_init = mu_init.copy()

            cluster = Cluster(
                event_num=self.v,
                id=i,
                alpha_cluster=self.alpha_clusters[i],
                event_names=self.event_names,
                decay=self.decay,
                Gao_para=self.Gau_para,
                beta_h=self.beta
            )
            cluster.alpha_k = alpha_init
            cluster.sigma_exp = sigma_init
            cluster.mu_k = mu_init
            cluster.beta_Rali = beta_init

            self.cluster_list.append(cluster)

    def cal_cache(self):
        for time_slot in self.time_slots:
            time_slot.cal_cache()

    def update_cluster(self):
        self.cluster_list = [cluster for cluster in self.cluster_list if len(cluster.time_slots) > 0]
        self.n_clusters = len(self.cluster_list)


    def update_time_slots(self, time_slot_length, Z):

        num_time_slots = int(self.T/ time_slot_length)
        for i in range(num_time_slots):
            cluster_id = np.where(Z[i] == 1)[0][0]
            self.time_slots[i].in_Cluster = cluster_id


    def generate_time_slots(self, time_slot_length, Z, beta):
        num_time_slots = int(self.T/ time_slot_length)

        time_slots = []

        for i in range(num_time_slots):
            if i==0:
                start_time = 0
            else:
                start_time =  i * time_slot_length
            end_time = start_time + time_slot_length
            cluster_id = np.where(Z[i] == 1)[0][0]
            time_slot = TimeSlot(start_time, end_time, cluster_id, self.event_names, self.decay, self.Gau_para, beta)
            time_slots.append(time_slot)

        for time_slot in time_slots:
            for index, row in self.event_table.iterrows():
                event_time = row['Start Time Stamp']
                if time_slot.start_time < event_time and event_time <= time_slot.end_time:
                    time_slot.events.append(row)
                    time_slot.event_count += 1

        return time_slots


    def initialize_Z(self, num_sequences, num_clusters):
        Z = np.zeros((num_sequences, num_clusters), dtype=int)
        for i in range(num_sequences):
            cluster_index = np.random.randint(0, num_clusters)
            Z[i, cluster_index] = 1
        return Z

    def get_event_table(self, event_table: pd.DataFrame):
        event_table = event_table.copy()
        event_table.columns = ['Node', 'Start Time Stamp', 'Event Name']

        event_table['Times'] = np.zeros(len(event_table))
        event_table = event_table.groupby(['Node', 'Start Time Stamp', 'Event Name']).count().reset_index()

        event_ind = event_table['Event Name'].astype('category')
        event_table['Event Ind'] = event_ind.cat.codes
        event_names = event_ind.cat.categories

        event_table.sort_values(['Node', 'Start Time Stamp', 'Event Ind'])
        return event_table, event_names


    def calculate_log_rho_nk_single(self, cluster: Cluster, time_slot: TimeSlot):

        E_log_pi_k = digamma(cluster.alpha_0) - digamma(np.sum(self.alpha_clusters))
        all_events_in_time_slot = time_slot.rou_nk_Q_lower_bound(cluster)
        log_rho_nk = E_log_pi_k + all_events_in_time_slot

        return log_rho_nk

    def update_r_nk_matrix(self):
        r_nk_threshold = 0.8
        r_nk_matrix = np.zeros((len(self.time_slots), len(self.cluster_list)))
        log_rho_nk = np.ones((len(self.time_slots),len(self.cluster_list)))
        for time_slot_index,time_slot in enumerate(self.time_slots):
            for cluster_index,cluster in enumerate(self.cluster_list):
                log_rho_nk[time_slot_index, cluster_index] = self.calculate_log_rho_nk_single(cluster,time_slot)

        for time_slot_index in range(len(self.time_slots)):
            sum_log_rho_nk = logsumexp(log_rho_nk[time_slot_index, :])
            r_nk_for_time_slot_index = np.exp(log_rho_nk[time_slot_index, :] - sum_log_rho_nk)
            r_nk_matrix[time_slot_index, :] = r_nk_for_time_slot_index

        for time_slot_index in range(len(self.time_slots)):
            max_index = np.argmax(r_nk_matrix[time_slot_index, :])
            if r_nk_matrix[time_slot_index, max_index] > r_nk_threshold:
                r_nk_matrix[time_slot_index, :] = 0
                r_nk_matrix[time_slot_index, max_index] = 1

        if np.array_equal(self.last_matrix, np.zeros((len(self.time_slots), len(self.cluster_list)))):
            self.last_matrix = r_nk_matrix.copy()
        if not np.array_equal(r_nk_matrix, self.last_matrix):
            r_nk_matrix = copy.deepcopy(self.last_matrix)
        return r_nk_matrix

    def update_alpha_0_k(self, cluster_index, r_nk_matrix):

        N_k = np.sum(r_nk_matrix[:, cluster_index])
        self.cluster_list[cluster_index].alpha_0 = self.cluster_list[cluster_index].alpha_0 + N_k
        for cluster_index in range(len(self.cluster_list)):
            self.alpha_clusters[cluster_index] = self.cluster_list[cluster_index].alpha_0

    def calculate_expected_pi_k(self):

        alpha_0_list = [cluster.alpha_0 for cluster in self.cluster_list]

        total_alpha_0 = np.sum(alpha_0_list)

        expected_pi_k_list = [alpha_0 / total_alpha_0 for alpha_0 in alpha_0_list]

        return expected_pi_k_list



    def Variational_EM(self,edge_mat):
        diff_threshold = 0.1
        Q_lower_bound_out_old = 0
        while True:

            r_nk_matrix = self.update_r_nk_matrix()
            self.r_nk_matrix = r_nk_matrix

            for cluster_index, cluster in enumerate(self.cluster_list):
                self.update_alpha_0_k(cluster_index, r_nk_matrix)
            expected_pi_k_list = self.calculate_expected_pi_k()
            for cluster_index, pi in enumerate(expected_pi_k_list):
                self.cluster_list[cluster_index].pi = pi

            Q_lower_bound_in_old= self.update_all_para(r_nk_matrix,edge_mat)
            while True:

                Q_lower_bound_in_new = self.update_all_para(r_nk_matrix, edge_mat)

                if(np.abs(Q_lower_bound_in_new-Q_lower_bound_in_old) < diff_threshold):
                    break
                Q_lower_bound_in_old = Q_lower_bound_in_new

            if (np.abs(Q_lower_bound_in_old - Q_lower_bound_out_old) < diff_threshold):
                break
            Q_lower_bound_out_old = Q_lower_bound_in_old

        return Q_lower_bound_out_old

    def update_all_para(self,r_nk_matrix,edge_mat):
        para_list = {}
        Q_lower_bound = 0
        alpha_mu_list = {}

        first_term = 0.0
        for v in range(self.v):
            for cluster_index, cluster in enumerate(self.cluster_list):
                if cluster.mu_k[v]==0 or cluster.beta_Rali[v]==0 :
                    first_term = 0
                else:
                    log_mu_k_v = np.log(cluster.mu_k[v])
                    term1 = log_mu_k_v
                    term2 = -0.5 * ((cluster.mu_k[v] / cluster.beta_Rali[v]) ** 2)
                    first_term += term1 + term2

        second_term = 0.0
        for cluster_index, cluster in enumerate(self.cluster_list):
            for v in range(self.v):
                for v_prime in range(self.v):
                    if cluster.sigma_exp[v][v_prime] == 0 or cluster.alpha_k[v][v_prime] == 0:
                        term = 0
                    else:
                        term = cluster.alpha_k[v][v_prime] / cluster.sigma_exp[v][v_prime]
                    second_term += term


        Q_lower_bound += first_term + second_term
        for time_slot_index,time_slot in enumerate(self.time_slots):
            para_list.setdefault(time_slot, {})
            for cluster_index, cluster in enumerate(self.cluster_list):
               L,alpha,mu,mu_q_ii_k,alpha_qijm_k = time_slot.EM_para_for_cluster(edge_mat,cluster)
               para_list[time_slot].setdefault(cluster, {})
               para_list[time_slot][cluster] = {
                   'L': L,
                   'mu_q_ii_k': mu_q_ii_k,
                   'alpha_qijm_k_upper_lower': alpha_qijm_k[0],
               }
               alpha_mu_list[time_slot]={'alpha': alpha, 'mu': mu,}
        for cluster_index, cluster in enumerate(self.cluster_list):
            b = 0
            c_sum = np.zeros(len(self.event_names))
            alpha_numerator = np.zeros([len(self.event_names), len(self.event_names)], dtype=object)
            alpha_denominator = np.zeros([len(self.event_names), len(self.event_names)], dtype=object)
            for n, time_slot in enumerate(self.time_slots):
                r_nk = r_nk_matrix[n,cluster.id]
                if r_nk == 0.0:
                    Q_lower_bound += 0
                    continue
                else:
                    Q_lower_bound += r_nk*para_list[time_slot][cluster]['L']
                b += r_nk*time_slot.T_n
                for v in cluster.event_names:
                    c_sum[v] += r_nk*para_list[time_slot][cluster]['mu_q_ii_k'][v]
                for v_i_index, v_i in enumerate(self.event_names):
                    for v_j_index,v_j in enumerate(self.event_names):
                        if para_list[time_slot][cluster]['alpha_qijm_k_upper_lower'][v_j][v_i] == 0:
                            upper = 0
                            lower = 0
                        else:
                            upper,lower = para_list[time_slot][cluster]['alpha_qijm_k_upper_lower'][v_j][v_i]
                        alpha_numerator[v_j][v_i] += r_nk*upper
                        alpha_denominator[v_j][v_i] += r_nk*lower
            for v_index, v in enumerate(self.event_names):
                a = 1 / (cluster.beta_Rali[v]) ** 2
                c = -1 - c_sum[v]
                b = b
                discriminant = b ** 2 - 4 * a * c
                if discriminant < 0:
                    raise ValueError("Discriminant is negative. No real roots.")
                sqrt_discriminant = np.sqrt(discriminant)
                if c_sum[v]==0 or a==0:
                    cluster.mu_k[v]=0
                    cluster.mu_k[v] = (-b + sqrt_discriminant) / (2 * a)
                cluster.beta_Rali[v] = np.sqrt(2 / np.pi) * cluster.mu_k[v]

                for v_j in range(len(cluster.alpha_k[v])):
                    sigma_vi_vj_m_k = cluster.sigma_exp[v_j, v]
                    if sigma_vi_vj_m_k == 0:
                        denominator_second = np.inf
                    elif sigma_vi_vj_m_k < 1e-10:

                        denominator_second = 1e10
                    else:
                        denominator_second = 1 / sigma_vi_vj_m_k
                    if(alpha_denominator[v_j, v] == 0 or alpha_numerator[v_j, v]==0):
                        cluster.alpha_k[v_j, v] = 0
                    else:
                        cluster.alpha_k[v_j, v] = alpha_numerator[v_j, v] / (alpha_denominator[v_j, v]+denominator_second)
                    cluster.sigma_exp[v_j, v] = cluster.alpha_k[v_j, v]

        return Q_lower_bound

    def one_step_change_iterator(self, edge_mat):
        return map(lambda e: self.one_step_change(edge_mat, e),
                   product(range(len(self.event_names)), range(len(self.event_names))))

    def one_step_change(self, edge_mat, e):
        j, i = e
        if (j == i):
            return edge_mat
        new_edge_mat = edge_mat.copy()

        if (new_edge_mat[j, i] == 1):
            new_edge_mat[j, i] = 0
            return new_edge_mat
        else:
            new_edge_mat[j, i] = 1
            new_edge_mat[i, j] = 0
            return new_edge_mat

    def Hill_Climb(self):

        edge_mat = self.init_structure - np.diag(self.init_structure.diagonal()) + np.eye(self.v, self.v)
        for cluster_index, cluster in enumerate(self.cluster_list):
            cluster.beta_h = edge_mat
        result = self.Variational_EM(edge_mat)
        L = result
        while (1):
            stop_tag = True
            for new_edge_mat in (list(self.one_step_change_iterator(edge_mat))):

                self.alpha_clusters = [5] * self.n_clusters
                self.cluster_list = []
                self.init_clusters()
                self.cluster_pi_list = np.random.dirichlet(self.alpha_clusters, size=1).flatten()
                for index, cluster in enumerate(self.cluster_list):
                    cluster.pi = self.cluster_pi_list[index]

                self.num_time_slots = int(self.T / self.time_slot_length)

                self.Z = self.initialize_Z(self.num_time_slots, self.n_clusters)
                for cluster_index, cluster in enumerate(self.cluster_list):
                    cluster.beta_h = edge_mat

                new_result = self.Variational_EM(new_edge_mat)

                new_L = new_result
                if (new_L > L):
                    result = new_result
                    L = new_L
                    stop_tag = False
                    edge_mat = new_edge_mat

            if (stop_tag):
                return result, edge_mat


from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def get_performance(adj, true_adj, threshold, draw=False, name='test', drop_diag=True):
    print(name + ':')
    if (drop_diag):
        adj = adj - np.diag(np.diag(adj))
        true_adj = true_adj - np.diag(np.diag(true_adj))
    if (draw):
        f, (ax1, ax2, ax3) = plt.subplots(figsize=(17, 4), ncols=3)
        sns.heatmap(adj, ax=ax1)
        sns.heatmap(adj > threshold, ax=ax2)
        sns.heatmap(true_adj, ax=ax3)
        plt.savefig(name)
        plt.show()
    precision = metrics.precision_score(true_adj.ravel(), adj.ravel() > threshold)
    recall = metrics.recall_score(true_adj.ravel(), adj.ravel() > threshold)
    f1 = metrics.f1_score(true_adj.ravel(), adj.ravel() > threshold)
    print(f'precision:{precision}\n,recall:{recall}\n,f1:{f1}\n')
    return {'recall': recall, 'precision': precision, 'f1': f1}


def generate_parameter_ranges(alpha_range: Tuple[float, float],
                              mu_range: Tuple[float, float],
                              hawkes_num: int,
                              alpha_margin: float) -> Dict[str, Dict[str, Tuple[float, float]]]:
    parameter_ranges = {}
    alpha_min, alpha_max = alpha_range

    for i in range(hawkes_num):
        range_start = alpha_min + i * (alpha_max - alpha_min) / hawkes_num
        range_end = alpha_min + (i + 1) * (alpha_max - alpha_min) / hawkes_num

        adjusted_start = range_start + alpha_margin * i
        adjusted_end = range_end - alpha_margin * (hawkes_num - 1 - i)

        alpha_range_i = (max(alpha_min, adjusted_start), min(alpha_max, adjusted_end))
        mu_range_i = mu_range

        parameter_ranges[f"cluster_{i + 1}"] = {
            'alpha_range': alpha_range_i,
            'mu_range': mu_range_i
        }

    return parameter_ranges

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="data")
    parser.add_argument('-n', '--event_num', type=int, help='传入事件种类数目', default=9)
    parser.add_argument('-ss', '--sample_size', type=int, help='传入样本大小', default=15000)
    parser.add_argument('-od', '--out_degree_rate', type=float, help='传入出度rate', default=1.5)
    parser.add_argument('-ar_0', '--alpha_range', type=str, help='传入节点数目', default='0.02,0.05')
    parser.add_argument('-mr_0', '--mu_range', type=str, help='传入节点数目', default='0.00003,0.00005')
    parser.add_argument('-t', '--Time_range', type=int, help='传入设备时间长度', default=38640)
    parser.add_argument('-ind', '--Exp_index', type=int, help='传入实验编号', default=0)
    parser.add_argument('-tsl', '--time_slot_len', type=int, help='时间片长度', default=432000)
    parser.add_argument('-hn', '--hawkes_num', type=int, help='初始霍克斯数量', default=2)
    parser.add_argument('-d', '--decay', type=int, help='指数核函数参数', default=0.1)
    parser.add_argument('-dta', '--delta', type=int, help='离散时间区间长度', default=1)
    parser.add_argument('-mean', '--mean_len', type=int, help='核函数平均影响时间', default=10)
    parser.add_argument('-rr', '--remove_rate', type=int, help='移除边的比例', default=0.2)
    args = parser.parse_args()
    delta = args.delta
    loc = args.mean_len
    hawkes_num = args.hawkes_num
    n = args.event_num
    decay = args.decay
    sample_size = args.sample_size
    out_degree_rate = args.out_degree_rate
    time_slot_len = args.time_slot_len
    Exp_index = args.Exp_index
    remove_rate = args.remove_rate

    alpha_range = tuple([float(i) for i in args.alpha_range.split(',')])
    mu_range = tuple([float(i) for i in args.mu_range.split(',')])

    alpha_margin = 0.002
    alpha_min, alpha_max = alpha_range
    mu_min, mu_max = mu_range

    parameter_ranges = generate_parameter_ranges(
        alpha_range=(alpha_min, alpha_max),
        mu_range=(mu_min, mu_max),
        hawkes_num=hawkes_num,
        alpha_margin=alpha_margin
    )
    file_name = f'×××'
    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
    target_path = os.path.join(parent_directory, 'uniform')
    file_path = os.path.join(target_path, file_name)

    event_table_ori, event_table_merge, edge_mat, topo, alpha_lists = pickle.load(
        open(file_path,'rb'))
    event_table_ori.columns = ['Node', 'Start Time Stamp', 'End Time Stamp', 'Event Name', 'Times']
    event_table = event_table_ori[['Node', 'Start Time Stamp', 'Event Name']]
    alpha_1 = alpha_lists[0]
    binary_alpha_1 = (alpha_1 != 0).astype(int)
    Gau_para = (0.5, 0.1)
    NSNID_HP_para ={
        'event_table': event_table,
        'decay':decay,
        'Gau_para':Gau_para,
    }
    np.seterr(divide='ignore', invalid='ignore')

    beta_h_0 = edge_mat

    obj = Simulation_NSNID_HP(event_table,decay,Gau_para,beta_h_0,alpha_range=alpha_range,
                              parameter_ranges=parameter_ranges,n_hawkes=hawkes_num,kernel_m=1,time_slot_length=time_slot_len)

    result,edge_mat_final = obj.Hill_Climb()

    threshold = 0.01
    results = []

    for idx in range(hawkes_num):
        true_alpha = alpha_lists[idx][0]
        binary_true = (true_alpha != 0).astype(int)
        est_alpha = obj.cluster_list[idx].alpha_k
        binary_est = np.where(est_alpha < threshold, 0, 1)
        perf = get_performance(
            binary_est,
            binary_true,
            0.00,
            name=f'edge_mat_cluster_{idx + 1}'
        )
        results.append(perf)

    perf_all = get_performance(edge_mat_final, edge_mat, 0.00, name='edge_mat_all')
    results.append(perf_all)


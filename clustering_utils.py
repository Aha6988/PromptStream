from scipy.optimize import linear_sum_assignment
import numpy as np
import datetime
import math
import pandas as pd
from collections import Counter, deque
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

#The 3-Clause BSD License
# For Priberam Clustering Software
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. ("PRIBERAM") (www.priberam.com)
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder (PRIBERAM) nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cluster_mapping(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1
    print(count_matrix)
    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    #accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment #, accuracy


def sparse_dotprod(fv0, fv1):
    dotprod = 0

    for f_id_0, f_value_0 in fv0.items():
        if f_id_0 in fv1:
            f_value_1 = fv1[f_id_0]
            dotprod += f_value_0 * f_value_1

    return dotprod


def cosine_bof(d0, d1):
    cosine_bof_v = {}
    for fn, fv0 in d0.items():
        if fn in d1:
            fv1 = d1[fn]
            if fv0 != 0 and fv1 != 0:
                cosine_bof_v[fn] = sparse_dotprod(
                    fv0, fv1) / math.sqrt(sparse_dotprod(fv0, fv0) * sparse_dotprod(fv1, fv1))
    return cosine_bof_v


def normalized_gaussian(mean, stddev, x):
  return (math.exp(-((x - mean) * (x - mean)) / (2 * stddev * stddev)))


def timestamp_feature(tsi, tst, gstddev):
  return normalized_gaussian(0, gstddev, (tsi-tst)/(60*60*24.0))


def sim_bof_dc(d0, c1): 
    numdays_stddev = 3.0
    bof = cosine_bof(d0.reprs, c1.reprs)
    bof["NEWEST_TS"] = timestamp_feature(
        d0.timestamp.timestamp(), c1.newest_timestamp.timestamp(), numdays_stddev)
    bof["OLDEST_TS"] = timestamp_feature(
        d0.timestamp.timestamp(), c1.oldest_timestamp.timestamp(), numdays_stddev)
    bof["RELEVANCE_TS"] = timestamp_feature(
        d0.timestamp.timestamp(), c1.get_relevance_stamp(), numdays_stddev)
    bof["ZZINVCLUSTER_SIZE"] = 1.0 / float(100 if c1.num_docs > 100 else c1.num_docs)
    return bof


def sim_bof_cc(c0, c1): 
    bof = cosine_bof(c0.reprs, c1.reprs)
    bof["dens_reps"] = cosine_similarity(c0.get_dens_rep().reshape(1, -1), c1.get_dens_rep().reshape(1, -1))
    return bof


def sim_dc(d0, c1): 
    sim = cosine_similarity(d0.dens_vec.reshape(1, -1), c1.get_dens_rep().reshape(1, -1))
    return sim


class Document:
    def __init__(self, archive_document):
        self.id = archive_document["id"]
        #self.reprs = archive_document["features"]
        self.dens_vec = np.array(archive_document["doc_vec"])
        self.timestamp = archive_document["date"]


class OnlineCluster:
    def __init__(self, document: Document):
        self.ids = set()
        self.num_docs = 0
        self.dens_vec = None
        self.add_document(document)
        self.sim_avg = AverageMeter()
        self.sim_thr = 1

    def add_document(self, document: Document):
        self.ids.add(document.id)
        self.num_docs = len(self.ids)
        self.__add_dens_vec(document.dens_vec)

    def __add_dens_vec(self, dens_vec):
        if self.dens_vec is None:
            self.dens_vec = dens_vec
        else:
            self.dens_vec = np.add(self.dens_vec, dens_vec)           

    def get_dens_rep(self):
        return self.dens_vec / self.num_docs


class Cluster:
    def __init__(self, documents: pd.DataFrame, local_label: int, global_label: int):
        self.local_label = local_label
        self.global_label = global_label
        self.ids = set()
        self.num_docs = 0
        self.reprs = {}
        self.dens_vec = np.zeros(1024)
        self.sum_timestamp = 0
        self.sumsq_timestamp = 0
        self.newest_timestamp = datetime.datetime.strptime(
            "1000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.oldest_timestamp = datetime.datetime.strptime(
            "3000-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        self.add_documents(documents)

    def get_relevance_stamp(self):
        z_score = 0
        mean = self.sum_timestamp / self.num_docs
        try:
          std_dev = math.sqrt((self.sumsq_timestamp / self.num_docs) - (mean*mean))
        except:
          std_dev = 0.0
        return mean + ((z_score * std_dev) * 3600.0) # its in secods since epoch
    
    def add_documents(self, documents: pd.DataFrame):
        for index, row in documents.iterrows():
            self.add_document(Document(row))
            
    def add_document(self, document: Document):
        self.ids.add(document.id)
        self.num_docs = len(self.ids)
        self.newest_timestamp = max(self.newest_timestamp, document.timestamp)
        self.oldest_timestamp = min(self.oldest_timestamp, document.timestamp)
        ts_hours =  (document.timestamp.timestamp() / 3600.0)
        self.sum_timestamp += ts_hours
        self.sumsq_timestamp += ts_hours * ts_hours
        self.__add_bof(document.reprs)
        self.__add_dens_vec(document.dens_vec)

    def __add_bof(self, reprs0):
        for fn, fv0 in reprs0.items():
            if fv0 is not None:
                if fn in self.reprs:
                    for f_id_0, f_value_0 in fv0.items():
                        if f_id_0 in self.reprs[fn]:
                            self.reprs[fn][f_id_0] += f_value_0
                        else:
                            self.reprs[fn][f_id_0] = f_value_0
                else:
                    self.reprs[fn] = fv0
        self.num_docs += 1

    def __add_dens_vec(self, dens_vec):
        self.dens_vec = np.add(self.dens_vec, dens_vec)

    def get_dens_rep(self):
        return self.dens_vec / self.num_docs


class Aggregator:
    def __init__(self, mapping_thr=0.5, similarity_thr=0.5, q_size=7): #,  model: model.Model, thr, merge_model: model.Model = None):
        self.clusters = []
        self.latest_clusters = deque(maxlen=q_size)
        #self.clusters.append(Cluster(pd.DataFrame(), -1, -1))
        self.mapping_thr = mapping_thr
        self.similarity_thr = similarity_thr
        self.max_label = -1

    def add_documents_to_clusters(self, documents, global_labels):
            documents["global_label"] = global_labels
            for i in list(set(global_labels)):
                cluster_df = documents[documents['global_label'] == i]
                for cluster in self.clusters:
                    if cluster.global_label == i:
                        cluster.add_documents(cluster_df)

    def add_documents_to_cluster(self, documents, global_label):
            for cluster in self.clusters:
                if cluster.global_label == global_label:
                    cluster.add_documents(documents)
        
    def make_new_clsuter(self, documents, local_label, global_label):
        new_cluster = Cluster(documents, local_label, global_label)
        self.clusters.append(new_cluster)

    def find_cluster(self, global_label):
        for clus in self.clusters:
            if clus.global_label == global_label:
                return clus
        return None

    def print_cluster_pool(self):
        for clus in self.clusters:
            print('{} : {}'.format(clus.global_label, clus.num_docs))

    #predicts clusters for the latest samples including new batch and part of previous model data
    def mapper(self, sample_df, sample_new_preds, sample_old_preds, batch_size):
        CLUSTER_MAP = {}
        sample_df["new_pred"] = sample_new_preds
        sample_df["old_pred"] = sample_old_preds
        new_batch_df = sample_df.tail(batch_size)
        unassigned_clusters = []
        new_clusters = {}
        for i in list(set(sample_new_preds)):
            cluster_df = sample_df[sample_df['new_pred'] == i]
            new_clusters[i] = Cluster(cluster_df, i, -1)
            counts = Counter(cluster_df['old_pred'].to_list())
            potential_mapping = counts.most_common(1)[0][0]
            print('new cluster: {} potential mapping: {}'.format(i, potential_mapping))
            if potential_mapping == -1:
                if len(new_batch_df[new_batch_df['new_pred']==i]) == 0:
                    CLUSTER_MAP[i] = -1
                else:
                    unassigned_clusters.append(i)
                continue
            thr_count = int(len(cluster_df) * self.mapping_thr)
            overlap = counts[potential_mapping]
            print('cluster size', len(cluster_df))
            print('potent cluster part: ', overlap)
            old_global_cluster = self.find_cluster(potential_mapping)
            bof = sim_bof_cc(new_clusters[i], old_global_cluster)
            print('similarities between two clusters: ', bof)
            similarity_score = sum(bof.values())/len(bof)
            print('similarity score: ', similarity_score)
            #if (overlap > thr_count and similarity_score > self.similarity_thr) \
            #    or (len(new_batch_df[new_batch_df['new_pred']== i]) == 0):  ## Could be removed
            if (similarity_score > self.similarity_thr) \
                or (len(new_batch_df[new_batch_df['new_pred']== i]) == 0):
                print('Mapping Accepted')
                CLUSTER_MAP[i] = potential_mapping
                self.add_documents_to_cluster(new_batch_df[new_batch_df['new_pred']==i], potential_mapping)
            else:
                unassigned_clusters.append(i)
            print('================================')
        for i in unassigned_clusters:
            if i != -1:
                self.max_label += 1
                CLUSTER_MAP[i] = self.max_label
                self.make_new_clsuter(new_batch_df[new_batch_df['new_pred']==i], i, self.max_label)
        CLUSTER_MAP[-1] = -1
        print('map: ', CLUSTER_MAP)
        return CLUSTER_MAP
            
        # potential_mapping = cluster_mapping(sample_old_preds, sample_new_preds)
        # for key, value in zip(potential_mapping.keys(), potential_mapping.values()):
        #     old_global_cluster = self.find_cluster(value)
        #     if old_global_cluster is not None:
        #         print('key: {} value: {}'.format(key, value))
        #         # cluster_df = sample_df[sample_df['local_pred'] == key]
        #         # print(cluster_df.old_pred)
        #         # confirm_num = int(sample_old_preds.tolist().count(value) * mapping_thr)
        #         # if sample_new_preds.tolist().count(key) > confirm_num:
        #         #     CLUSTER_MAP[key] = value
        #         confirm_num = int(sample_new_preds.tolist().count(key) * self.mapping_thr)
        #         shared_list = [i for i in range(len(sample_df)) if (sample_old_preds[i] == value and sample_new_preds[i] == key)]
        #         print('cluster size', sample_new_preds.tolist().count(key))
        #         print('old part: ', len(shared_list))
        #         bof = sim_bof_cc(new_clusters[key], old_global_cluster)
        #         # print('similarities between two clusters: ', bof)
        #         similarity_score = sum(bof.values())/len(bof)
        #         print('similarity score: ', similarity_score)
        #         if len(shared_list) > confirm_num and similarity_score > self.similarity_thr:
        #             CLUSTER_MAP[key] = value
        # return CLUSTER_MAP
        
    # predicts clusters for the new batch, considerig only clusters in the window for mapping 
    def sim_mapper(self, new_df, new_preds):
        CLUSTER_MAP = {}
        new_df["new_pred"] = new_preds
        new_clusters = {}
        unassigned_clusters = []
        latest_clus = list(self.latest_clusters)
        latest_flat = [item for sublist in latest_clus for item in sublist]
        latest_flat = list(set(latest_flat))
        print('clusters in the window: ', latest_flat)
        for i in list(set(new_preds)): # i is the cluster local label
            cluster_df = new_df[new_df['new_pred'] == i]
            new_clusters[i] = Cluster(cluster_df, i, -1)
            best_match = -1
            best_sim = 0.0
            for c_id in latest_flat:
                potential_c = self.find_cluster(c_id)
                bof = sim_bof_cc(new_clusters[i], potential_c)
                #print(bof)
                similarity_score = sum(bof.values())/len(bof)
                #print('similarity score: ', similarity_score)
                if similarity_score > best_sim and similarity_score > self.similarity_thr:
                    best_sim = similarity_score
                    best_match = c_id
            #print('====================================')
            if best_match == -1:
                unassigned_clusters.append(i)
            else:
                CLUSTER_MAP[i] = best_match

        for new_c, old_c in CLUSTER_MAP.items():
            old_global_cluster = self.find_cluster(old_c)
            old_global_cluster.add_documents(new_df[new_df['new_pred'] == new_c])

        for new_c in unassigned_clusters:
            self.max_label += 1
            CLUSTER_MAP[new_c] = self.max_label
            new_clusters[new_c].global_label = self.max_label
            self.clusters.append(new_clusters[new_c])

        return CLUSTER_MAP
    

class OnlineAggregator:
    def __init__(self, similarity_thr=0.5, q_size=7): #,  model: model.Model, thr, merge_model: model.Model = None):
        self.clusters = [] #indices are cluster ids
        self.curr_clusters = set()
        self.latest_clusters = deque(maxlen=q_size)
        self.similarity_thr = similarity_thr

    def print_cluster_pool(self):
        for i in range(len(self.clusters)):
            print('{} : {}'.format(i, self.clusters[i].num_docs))

    #assigns a document to a cluster for online clustering
    def put_document(self, document: pd.DataFrame):
        doc = Document(document)
        latest_clus = list(self.latest_clusters)
        #print('clusters in the recent days: ', latest_clus)
        #print('clusters in today\'s data: ', list(self.curr_clusters))
        window_flat = list(set([item for sublist in latest_clus for item in sublist] + list(self.curr_clusters)))
        best_match = -1
        best_sim = 0.0
        max_sim = 0.0
        for c_id in window_flat:
            #potential_c = self.find_cluster(c_id)
            potential_c = self.clusters[c_id]
            sim = sim_dc(doc, potential_c)
            #print(f'cluster: {c_id} sim: {sim}')
            if sim > max_sim:
                max_sim = sim[0][0]
            if sim > best_sim and sim > self.similarity_thr:
            #if sim > potential_c.sim_thr and sim > best_sim:
                best_sim = sim[0][0]
                best_match = c_id
        #print('----------------------')
        if best_match == -1:
            self.clusters.append(OnlineCluster(doc))
            #self.clusters[-1].sim_thr = self.similarity_thr
            best_match = len(self.clusters)-1
            best_sim = 1 - max_sim
        else:
            self.clusters[best_match].add_document(doc)
            #self.clusters[best_match].sim_avg.update(best_sim, 1)
            #self.clusters[best_match].sim_thr = max(self.clusters[best_match].sim_avg.avg/2, self.similarity_thr)

        return best_match, best_sim




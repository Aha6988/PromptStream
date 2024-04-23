import torch
import torch.nn as nn
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from tqdm import trange, tqdm
from collections import deque, Counter
import random
import copy

from datastream_simulation import DailyDataStreamSimulation_df
from clustering_utils import OnlineAggregator
from encoders import PromptEncoder
import utils
import b3
import math


import warnings
warnings.filterwarnings("ignore")


class Config:
    #Online clustering 
    mean_similarity_thr = 0.5
    q_size = 2   #In fact we keep the clusters of "window_size = q_size + slide_size"
    slide_size = 1

    #dataset
    #The dataset needs to contain the columns [date(e.g.'2024-05-22'), title(str), body(str), cluster(int)]
    data_name = 'News14'
    CUSTOM_DATASET = './datasets/my_custom_dataset.json'

    #Model
    model_name = 'sentence-transformers/all-roberta-large-v1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_seq_len = 128

    #Initial training of prompt representations
    init_steps = 10 # for training prompt representations, in this period we use mean representations of documents
    init_batch = 64
    init_epochs = 3 # for initial training

    #Continual updating of prompt representations
    epochs = 1 # for training prompt representations continually with the examples in the memory 
    memory_size = 10 #please see the "Confidence-Aware Memory Replay" section in the paper (N in the paper)
    sample_thr = 0.5 #choose samples with confidence higher than this threshold to be stroed in memory
    temp = 0.2 #for contrastive loss
    lr = 5e-6
    pattern_id = 0 #prompt_template in "make_prompt" function
    min_cluster_size = 1

    #General
    random_seed = 53


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


def eval_metric(label, cluster): 
    ami = np.round(metrics.adjusted_mutual_info_score(label, cluster),3)
    ari = np.round(metrics.adjusted_rand_score(label, cluster),3)
    fscore, precision, recall = [np.round(k,3) for k in b3.calc_b3(label,cluster)]
    
    return [precision, recall, fscore, ami, ari]


def indices_uniform_sampling(N, cluster_indxs, cluster_sims, min_cluster_size):
    """Samples elements uniformely accross labels.
    Args:
        N (int): size of returned data.
        cluster_lists: dict of key (cluster), value (indexes of datapoints in this cluster)
    """
    nmb_non_empty_clusters = 0
    for i in range(len(cluster_indxs)):
        if len(cluster_indxs[i]) >= min_cluster_size:
            nmb_non_empty_clusters += 1

    size_per_cluster = int(N / nmb_non_empty_clusters) + 1
    res = np.array([])

    for i in range(len(cluster_indxs)):
        # skip empty clusters
        if len(cluster_indxs[i]) < min_cluster_size:
            continue
        indexes = np.random.choice(
            cluster_indxs[i],
            size_per_cluster,
            replace=(len(cluster_indxs[i]) <= size_per_cluster),
            p=softmax(cluster_sims[i])
        )
        res = np.concatenate((res, indexes))

    np.random.shuffle(res)
    res = list(res.astype('int'))
    if len(res) >= N:
        return res[:N]
    res += res[: (N - len(res))]
    return res


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#contrastive loss
def infonce_loss(sample_outputs, class_indices, class_embds, temp = 0.2):
    loss = 0
    for i in range(len(sample_outputs)):
        exp_temp_sims = torch.exp(torch.nn.functional.cosine_similarity(sample_outputs[i], class_embds)/temp)
        loss += -1*torch.log(exp_temp_sims[class_indices[i]]/torch.sum(exp_temp_sims))
    return loss


def load_data(conf):
    dataset = conf.data_name
    if dataset == 'WCEP19':
        dataset_df = pd.read_json('datasets/WCEP19_raw.json')
        dataset_df['cluster'] = dataset_df['story']
        data_stat(dataset_df)

    elif dataset == 'WCEP18':
        dataset_df = pd.read_json('datasets/WCEP18_raw.json')
        dataset_df['cluster'] = dataset_df['story']
        data_stat(dataset_df)

    elif dataset == 'News14':
        dataset_df = pd.read_json('datasets/News14_raw.json')
        dataset_df['cluster'] = dataset_df['story']
        data_stat(dataset_df)

    elif dataset == 'custom':
        dataset_df = pd.read_json(conf.CUSTOM_DATASET)
        data_stat(dataset_df)

    else:
        dataset_df = None
        print('You need to specify the dataset...')

    return dataset_df


def round_timedelta(td):
    """
    Rounds a timedelta object to the nearest day, and returns the number of days rounded to.
    """
    total_seconds = td.total_seconds()
    days = math.ceil(total_seconds/86400)
    return days


def data_stat(dataset_df):
    print(dataset_df.head())
    print('\n')
    print(np.min(dataset_df.date))
    print(np.max(dataset_df.date))
    clusters = dataset_df.groupby('cluster')
    cluster_labs = list(clusters.groups.keys())
    print(f'Dataset size: {len(dataset_df)} Number of clusters: {len(cluster_labs)}')
    cluster_duration = []
    cluster_size = []
    for cluster_lab in cluster_labs: 
        cluster = clusters.get_group(cluster_lab)
        delta = np.max(cluster.date) - np.min(cluster.date)
        cluster_duration.append(round_timedelta(delta)+1)
        cluster_size.append(len(cluster))
    print('mean cluster duration: ', np.mean(cluster_duration))
    print('min cluster duration: ', np.min(cluster_duration))
    print('max cluster duration: ', np.max(cluster_duration))
    print('median cluster duration: ', np.median(cluster_duration))

    print('mean number of articles in a cluster in a day: ', np.mean(np.array(cluster_size)/np.array(cluster_duration)))

    dataset_df['date2'] = dataset_df.date.dt.date
    per_day_data = dataset_df.groupby('date2')
    print('mean data/day size: ', np.mean(per_day_data.size()))
    print('min data/day size: ', np.min(per_day_data.size()))
    print('max data/day size: ', np.max(per_day_data.size()))


def make_prompt(title, body, pattern_id=0):
    if pattern_id == 0:
        prompt = " [ topic : <mask> ] " + title + " " + body

    elif pattern_id == 1: 
        prompt = "(This news is about: <mask>) " + title + " " + body

    elif pattern_id == 2: 
        prompt = " <mask> " + title + " " + body

    elif pattern_id == 3:
        prompt = "Keywords: <mask>" + '\n' + title + '\n' + body

    elif pattern_id == 4:
        prompt =  (title + '\n' + body)[0:250] + '\n' + "Keywords: <mask>"

    else: 
        prompt = " [ topic : <mask> ] " + body + " " + title

    return prompt
    

def main(data, conf):
    AGGREGATOR = OnlineAggregator(similarity_thr=conf.mean_similarity_thr, q_size=conf.q_size)
    DATA_STREAM = DailyDataStreamSimulation_df(data)
    STREAM_PREDS = []
    STREAM_SIMS = []
    TOKENIZER = AutoTokenizer.from_pretrained(conf.model_name)
    MODEL = PromptEncoder(conf, TOKENIZER.mask_token_id).to(conf.device)
    # MODEL = nn.DataParallel(MODEL).to(conf.device)

    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=conf.lr)

    #*************************** Training PromptEncoder **************************
    init_train_data = []
    MODEL.eval()
    for i in range(conf.init_steps):
        new_df = DATA_STREAM.pop_training_data(days=conf.slide_size)
        print('start date: ', np.min(new_df.date))
        print('end date: ', np.max(new_df.date))
        print('batch size: ', len(new_df))
        preds = []
        sims = []

        for indx, doc in new_df.iterrows():
            text = doc.title + ' ' + doc.text
            text_enc = TOKENIZER(text, return_tensors="pt", truncation=True) #, max_length=conf.max_seq_len)
            input_ids = text_enc.input_ids.to(conf.device)
            attention_mask = text_enc.attention_mask.to(conf.device)
            with torch.no_grad():
                doc["doc_vec"] = MODEL(input_ids=input_ids, attention_masks=attention_mask, task_type='mean').cpu().data.numpy()
            pred, sim = AGGREGATOR.put_document(doc)
            preds.append(pred)
            sims.append(sim)
            STREAM_PREDS.append(pred)
            STREAM_SIMS.append(sim)
            AGGREGATOR.curr_clusters.add(pred)
        AGGREGATOR.latest_clusters.append(list(AGGREGATOR.curr_clusters))
        print('clusters in the batch: ', list(AGGREGATOR.curr_clusters))
        AGGREGATOR.curr_clusters.clear()
        print('==========================')
        new_df['pred'] = preds #TODO
        new_df['sim'] = sims
        init_train_data.append(new_df)


    init_train_data_df = pd.concat(init_train_data, axis=0, ignore_index=True)
    utils.ScoreSet(init_train_data_df.cluster.values, STREAM_PREDS)
    fscore, precision, recall = [np.round(k,3) for k in b3.calc_b3(init_train_data_df.cluster.values, STREAM_PREDS)]
    print('bcubed scores: \n')
    print(f'precision: {precision}, recall: {recall}, F1: {fscore}')


    MODEL.train()


    print("Train data distribution: ", Counter(init_train_data_df[init_train_data_df.sim >= conf.sample_thr].pred.values))
    target_index = init_train_data_df[init_train_data_df.sim >= conf.sample_thr].index
    # sample_prob = softmax(window_data[window_data.sim >= conf.sample_thr].sim.values.astype(float))

    cluster_indxs = [[] for i in range(len(AGGREGATOR.clusters))]
    cluster_sims = [[] for i in range(len(AGGREGATOR.clusters))]
    for i in target_index:
        cluster_indxs[init_train_data_df.loc[i, 'pred']].append(i)
        cluster_sims[init_train_data_df.loc[i, 'pred']].append(init_train_data_df.loc[i, 'sim'])

    existing_clusters = list(range(len(AGGREGATOR.clusters)))
    class_embds = torch.tensor([c.get_dens_rep()[0] for c in AGGREGATOR.clusters], device=conf.device, requires_grad=False) #TODO
    num_itr = int(len(init_train_data_df)/conf.init_batch)+1
    for e in range(conf.init_epochs):
        MODEL.train()
        pbar = tqdm(range(num_itr))
        loss_avg = AverageMeter()
        for itr in range(num_itr):
            # samples = np.random.choice(target_index, conf.init_batch) #, p=sample_prob) #for w/o uniform sampler ablation study
            samples = indices_uniform_sampling(conf.init_batch, cluster_indxs, cluster_sims, conf.min_cluster_size)
            docs = init_train_data_df.loc[samples]

            prompts = [make_prompt(doc.text, doc.title, pattern_id=conf.pattern_id) for index, doc in docs.iterrows()]
            prompt_enc = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True, max_length=conf.max_seq_len)
            input_ids = prompt_enc.input_ids.to(conf.device)
            sample_embeddings = MODEL(input_ids=input_ids, task_type='prompt')

            class_indices = init_train_data_df.loc[samples, 'pred'].values

            sample_loss = infonce_loss(sample_embeddings, class_indices, class_embds, conf.temp)
            loss = sample_loss
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            loss_avg.update(loss.item())
            pbar.set_description(
                f"Epoch: {e+1}. "
                f"Iter: {itr+1}/{num_itr}. "
                f"Epoch-avg-loss: {loss_avg.avg:.4f}. "
                )
            pbar.update()
        pbar.close()


    
    AGGREGATOR = OnlineAggregator(similarity_thr=conf.mean_similarity_thr, q_size=conf.q_size)
    STREAM_PREDS = []
    STREAM_SIMS = []
    DATA_STREAM.reset_datastream()
    MODEL.eval()
    init_train_data = []
    for i in range(conf.init_steps):
        new_df = DATA_STREAM.pop_training_data(days=conf.slide_size)
        print('start date: ', np.min(new_df.date))
        print('end date: ', np.max(new_df.date))
        print('batch size: ', len(new_df))

        prompts = [make_prompt(doc.text, doc.title, pattern_id=conf.pattern_id) for indx, doc in new_df.iterrows()]
        prompts_enc = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True, max_length=conf.max_seq_len)
        input_ids = prompts_enc.input_ids.to(conf.device)
        with torch.no_grad():
            mask_rep = MODEL(input_ids=input_ids, task_type='prompt').cpu().data.numpy().tolist()


        text = [doc.title + ' ' + doc.text for index, doc in new_df.iterrows()] 
        text_enc = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=conf.max_seq_len)
        input_ids = text_enc.input_ids.to(conf.device)
        attention_mask = text_enc.attention_mask.to(conf.device)
        with torch.no_grad():
            mean_rep = MODEL(input_ids=input_ids, attention_masks=attention_mask, task_type='mean').cpu().data.numpy().tolist()

        comb_rep = np.add(mask_rep, mean_rep)
        new_df["doc_vec"] = comb_rep.tolist()

        for indx, doc in new_df.iterrows():
            pred, sim = AGGREGATOR.put_document(doc)
            STREAM_PREDS.append(pred)
            STREAM_SIMS.append(sim)
            AGGREGATOR.curr_clusters.add(pred)
        AGGREGATOR.latest_clusters.append(list(AGGREGATOR.curr_clusters))
        print('clusters in the batch: ', list(AGGREGATOR.curr_clusters))
        AGGREGATOR.curr_clusters.clear()
        print('==========================')
        init_train_data.append(new_df)

    init_train_data_df = pd.concat(init_train_data, axis=0, ignore_index=True)
    utils.ScoreSet(init_train_data_df.cluster.values, STREAM_PREDS)
    fscore, precision, recall = [np.round(k,3) for k in b3.calc_b3(init_train_data_df.cluster.values, STREAM_PREDS)]
    print('bcubed scores: \n')
    print(f'precision: {precision}, recall: {recall}, F1: {fscore}')
    print('==========================')

    #********************************* Slidign window ********************************
    tuned_ps, tuned_rs, tuned_f1s, tuned_amis, tuned_aris = [],[],[],[],[]
    new_df = DATA_STREAM.pop_training_data(days=conf.slide_size)
    begin_date = new_df.date.tolist()[0]
    counter = 1
    while new_df is not None:
        MODEL.eval()
        print('start date: ', np.min(new_df.date))
        print('end date: ', np.max(new_df.date))
        print('batch size: ', len(new_df))
        
        prompts = [make_prompt(doc.text, doc.title, pattern_id=conf.pattern_id) for index, doc in new_df.iterrows()]
        prompt_enc = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True, max_length=conf.max_seq_len)
        input_ids = prompt_enc.input_ids.to(conf.device)
        with torch.no_grad():
            mask_rep = MODEL(input_ids=input_ids, task_type='prompt').cpu().data.numpy().tolist()

        text = [doc.title + ' ' + doc.text for index, doc in new_df.iterrows()] 
        text_enc = TOKENIZER(text, return_tensors="pt", padding=True, truncation=True, max_length=conf.max_seq_len)
        input_ids = text_enc.input_ids.to(conf.device)
        attention_mask = text_enc.attention_mask.to(conf.device)
        with torch.no_grad():
            mean_rep = MODEL(input_ids=input_ids, attention_masks=attention_mask, task_type='mean').cpu().data.numpy().tolist()

        comb_rep = np.add(mask_rep, mean_rep)
        new_df["doc_vec"] = comb_rep.tolist()

        preds = []
        for indx, doc in new_df.iterrows():
            pred, sim = AGGREGATOR.put_document(doc)
            preds.append(pred)
            STREAM_PREDS.append(pred)
            STREAM_SIMS.append(sim)
            AGGREGATOR.curr_clusters.add(pred)
        AGGREGATOR.latest_clusters.append(list(AGGREGATOR.curr_clusters))
        print('clusters in the batch: ', list(AGGREGATOR.curr_clusters))
        AGGREGATOR.curr_clusters.clear()

        #Continual learning
        if counter % conf.memory_size == 0:
            MODEL.train()
            window_data = DATA_STREAM.pop_latest(conf.memory_size)
            window_data['pred'] = STREAM_PREDS[-len(window_data):]
            window_data['sim'] = STREAM_SIMS[-len(window_data):]
            target_index = window_data[window_data.sim >= conf.sample_thr].index
            # sample_prob = softmax(window_data[window_data.sim >= conf.sample_thr].sim.values.astype(float))
            print("Train data distribution: ", Counter(window_data[window_data.sim >= conf.sample_thr].pred.values))
            existing_clusters = window_data.pred.unique().astype(int).tolist()

            cluster_indxs = [[] for i in range(len(existing_clusters))]
            cluster_sims = [[] for i in range(len(existing_clusters))]
            for i in target_index:
                cluster_indxs[existing_clusters.index(window_data.loc[i, 'pred'])].append(i)
                cluster_sims[existing_clusters.index(window_data.loc[i, 'pred'])].append(window_data.loc[i, 'sim'])

            class_embds = torch.tensor([c.get_dens_rep() for c in np.array(AGGREGATOR.clusters)[existing_clusters]], device=conf.device, requires_grad=False)
            num_itr = int(len(window_data)/conf.init_batch)+1
            for e in range(conf.epochs):
                pbar = tqdm(range(num_itr))
                loss_avg = AverageMeter()
                for itr in range(num_itr):
                    # samples = np.random.choice(target_index, conf.init_batch) #, p=sample_prob)
                    samples = indices_uniform_sampling(conf.init_batch, cluster_indxs, cluster_sims, conf.min_cluster_size)
                    docs = window_data.loc[samples]
                    prompts = [make_prompt(doc.text, doc.title, pattern_id=conf.pattern_id) for index, doc in docs.iterrows()]
                    prompts_enc = TOKENIZER(prompts, return_tensors="pt", padding=True, truncation=True, max_length=conf.max_seq_len)
                    input_ids = prompts_enc.input_ids.to(conf.device)
                    sample_outputs = MODEL(input_ids, task_type='prompt')

                    class_indices = [existing_clusters.index(c) for c in window_data.loc[samples, 'pred']]

                    sample_loss = infonce_loss(sample_outputs, class_indices, class_embds, conf.temp)
                    loss = sample_loss
                    OPTIMIZER.zero_grad()
                    loss.backward()
                    OPTIMIZER.step()
                    loss_avg.update(loss.item())
                    pbar.set_description(
                        f"Epoch: {e+1}. "
                        f"Iter: {itr+1}/{num_itr}. "
                        f"Epoch-avg-loss: {loss_avg.avg:.4f}. "
                        )
                    pbar.update()
                pbar.close()


        eval_results = eval_metric(new_df.cluster, preds) #precision, recall, fscore, ami, ari        
        tuned_ps.append(np.round(eval_results[0],4))
        tuned_rs.append(np.round(eval_results[1],4))
        tuned_f1s.append(np.round(eval_results[2],4))
        tuned_amis.append(np.round(eval_results[3],4))
        tuned_aris.append(np.round(eval_results[4],4))

        print('==================================================================================')
        new_df = DATA_STREAM.pop_training_data(days=conf.slide_size)
        counter += 1


    # MODEL.encoder.save_pretrained('News14_p8_final_model')

    return STREAM_PREDS, begin_date, tuned_ps, tuned_rs, tuned_f1s, tuned_amis, tuned_aris


if __name__ == '__main__':
    conf = Config()
    data = load_data(conf)
    set_seed(conf.random_seed)
    stream_preds, begin_date, tuned_ps, tuned_rs, tuned_f1s, tuned_amis, tuned_aris = main(data, conf)
    print('window_averaged scores: ')
    print(f'begin date: {begin_date}. '
          f'B3-P: {np.round(np.mean(tuned_ps),4)}. '
          f'B3-R: {np.round(np.mean(tuned_rs),4)}. '
          f'B3-F: {np.round(np.mean(tuned_f1s),4)}. '
          f'AMI: {np.round(np.mean(tuned_amis),4)}. '
          f'ARI: {np.round(np.mean(tuned_aris),4)}. ')

    print('===================================')
    print('overla scores: ')
    print('scores: \n')
    utils.ScoreSet(data.cluster.values, stream_preds)
    fscore, precision, recall = [np.round(k,3) for k in b3.calc_b3(data.cluster.values, stream_preds)]
    print('bcubed scores: \n')
    print(f'precision: {precision}, recall: {recall}, F1: {fscore}')

    data['preds'] = stream_preds
    data = data[['id', 'date', 'title', 'text', 'cluster', 'preds']]
    data.to_csv('datasets/'+conf.data_name+'_pred.csv')
    

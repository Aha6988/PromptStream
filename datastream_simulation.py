import pandas as pd


class DailyDataStreamSimulation_df:
    daily_data_idxs: list[list[int]]
    dataset: pd.DataFrame
    n_days: int
    ds_counter: int

    def __init__(self, dataset_: pd.DataFrame):
        self.dataset = dataset_
        self.dataset['date'] = self.dataset.date.dt.date
        per_day_data = self.dataset.groupby('date')
        self.daily_data_idxs = []
        for group_name, group_members in per_day_data.groups.items():
            self.daily_data_idxs.append(list(group_members))
        self.n_days = len(self.daily_data_idxs)
        self.ds_counter = 0

    def reset_datastream(self):
        self.ds_counter = 0

    def pop_training_data(self, days=1):
        if self.ds_counter < len(self.daily_data_idxs):
            training_idxs = self.daily_data_idxs[self.ds_counter: min(self.ds_counter+days, len(self.daily_data_idxs))]
            flat_idxs = [item for sublist in training_idxs for item in sublist]
            training_df = self.dataset.loc[flat_idxs]
            self.ds_counter += days
        else:
            training_df = None
        return training_df
    
    def pop_latest(self, days=1):
        latest_idxs = self.daily_data_idxs[self.ds_counter-days:self.ds_counter]
        flat_idxs = [item for sublist in latest_idxs for item in sublist]
        latest_df = self.dataset.loc[flat_idxs]
        return latest_df


class DataStreamSimulation_df:
    dataset: pd.DataFrame
    ds_counter: int # the number of seen data points OR the pointer to the begining of the next batch
    upper_bound: int

    def __init__(self, dataset_: pd.DataFrame):
        self.dataset = dataset_
        self.ds_counter = 0
        self.upper_bound = len(dataset_)

    def dataset_sort(self, column_name):
        self.dataset = self.dataset.sort_values(column_name)

    def reset_datastream(self):
        self.ds_counter = 0

    def pop_training_data(self, size=500):
        training_df = self.dataset.iloc[self.ds_counter:self.ds_counter+size]
        self.ds_counter += size
        return training_df
    
    def pop_batch_data(self, size=500):
        if self.ds_counter < self.upper_bound:
            batch_df = self.dataset.iloc[self.ds_counter:min(self.ds_counter+size, self.upper_bound)]
            self.ds_counter = min(self.ds_counter+size, self.upper_bound)
        else:
            batch_df = None

        return batch_df
    
    def pop_single_data(self):
        if self.ds_counter < self.upper_bound:
            single_row_df = self.dataset.iloc[self.ds_counter]
            self.ds_counter += 1
        else:
            single_row_df = None   
        return single_row_df
    
    def pop_latest(self, nr_latest=500):
        latest_df = self.dataset.iloc[self.ds_counter-nr_latest:self.ds_counter]
        return latest_df
    


class DataStreamSimulation:
    dataset: pd.DataFrame
    ds_counter: int

    def __init__(self, dataset_: pd.DataFrame):
        self.dataset = dataset_
        self.ds_counter = 0

    def pop(self):
        text = self.dataset.iloc[self.ds_counter].body
        self.ds_counter += 1
        # if len(self.dataset) < self.ds_counter: 
        #     END or restart ?
        return text

    def reset_datastream(self):
        self.ds_counter = 0

    def dataset_sort(self, time_column):
        self.dataset = self.dataset.sort_values(time_column)

    def pop_training_data(self, size=2000):
        texts = self.dataset["body"].iloc[self.ds_counter:self.ds_counter+size].to_list()
        self.ds_counter += size
        return texts
    
    def pop_latest(self, nr_latest=1):
        texts = self.dataset["body"].iloc[self.ds_counter-nr_latest:self.ds_counter].to_list()
        return texts

# coding: utf-8

"""
this module creates data loader for model.
"""

import os
import sys
import torch
import torch.utils.data as data
import numpy as np

class Dataset(data.Dataset):
    """ Custom data.Dataset compatible with data.DataLoader """
    def __init__(self, src_path):
        """ read source file from txt file """
        self.src = open(src_path).readlines()
        self.num_total_sap = len(self.src)
        self.max_sym = 150
        self.max_men = 15
        self.max_sym_id = 9549
        self.max_men_id = 949
        self.max_ent_id = 587

    def __len__(self):
        return self.num_total_sap

    def __getitem__(self, index):
        """ return one data pair (symptom, mentions, entitys) """
        seg_list = self.src[index].split('\1')
        assert(len(seg_list) == 3)
        symptoms, mentions, entities = seg_list
        sym_lens = len(symptoms.split('\t'))
        men_lens = len(mentions.split('\t'))
        symptoms, mentions, entities = self.prepadding(symptoms, mentions, entities, sym_lens, men_lens)
        return symptoms, mentions, entities, sym_lens, men_lens

    def prepadding(self, symptoms, mentions, entities, sym_lens, men_lens):
        """ padding symptoms to maximum size p
            padding mentions and entities to maximum size q
        """
        entities = entities.strip()
        for i in range(self.max_sym - sym_lens):
            symptoms += '\t' + str(self.max_sym_id)
        for i in range(self.max_men - men_lens):
            mentions += '\t' + str(self.max_men_id)
        for i in range(self.max_men - men_lens):
            entities += '\t' + str(self.max_ent_id)

        symptoms = self.str2int(symptoms)
        mentions = self.str2int(mentions)
        entities = self.str2int(entities)
        return symptoms, mentions, entities
        #return symptoms.split('\t'), mentions.split('\t'), entities.split('\t')

    def str2int(self, str_):
        str = str_.split('\t')
        str = np.array(str).astype(np.int64)
        return str


def get_loader(file_path, batch_size=2):
    """ return data loader for cusdom dataset """

    # build a custom dataset
    dataset = Dataset(file_path)

    # data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    return data_loader


if __name__ == '__main__':
    src_path = '../../res/train_data_v3/train_data.txt'
    data_loader = get_loader(src_path)
    data_iter = iter(data_loader)
    symptoms, mentions, entities, sym_lens, men_lens = next(data_iter)
    print symptoms
    print entities
    print sym_lens, type(entities)
# coding: utf-8

import os
import sys
import numpy as np
import argparse
import pandas as pd
import cPickle as pickle

class DataProcess:
    """ preprocess train and test data """
    def __init__(self, src_path, vec_path, res_path, sym_vec_path, diag_vec_path, sym_pickle_path, diag_pickle_path):
        self.src_path = src_path
        self.vec_path = vec_path
        self.res_path = res_path
        self.sym_path = sym_vec_path
        self.diag_path = diag_vec_path
        self.sym_pickle = sym_pickle_path
        self.diag_pickle = diag_pickle_path

        self.word_embedding = {}
        self.sym_dict = {}
        self.men_dict = {}
        self.en_dict = {}

    def read_word_embedding(self):
        """ load word embedding """
        with open(self.vec_path, 'r') as f:
            line = f.readline()
            line = f.readline()  # skip column name
            while line:
                line = line.strip()
                word2id, embedding = line.split(' ', 1)
                self.word_embedding[word2id] = embedding
                line = f.readline()

    def convert2embedding(self, seg):
        """
        :param seg
        :return: embedding vector
        """
        vec = [0.0 for i in range(100)]
        for word in seg:
            if word not in self.word_embedding:
                continue
            word_embedding = self.word_embedding[word]
            word_embedding = [eval(i) for i in word_embedding.split()]
            vec = map(sum, zip(vec, word_embedding))
        return map(str, vec)

    def construct_dict(self, file_name, pickle_path, dict_name, idx, is_en, sep='\t'):
        """ construct dict and save in file_name """
        of = open(file_name, 'w')
        embedding = []
        with open(self.src_path, 'r') as f:
            line = f.readline()
            cnt = 0
            cnt1 = 0
            while line:
                line = line.strip()
                seg_list = line.split('@')
                if is_en == True:
                    en_list = seg_list[-1].split('\t')
                    for en in en_list:
                        if en not in self.en_dict:
                            self.en_dict[en] = cnt1
                            cnt1 += 1
                seg_list = seg_list[idx].split(sep)
                for seg in seg_list:
                    if seg not in dict_name:
                        dict_name[seg] = cnt
                        seg_embedding = self.convert2embedding(list(seg.decode('utf8')))
                        embedding.append(seg_embedding)
                        of.write(str(cnt) + '\t' + '\t'.join(seg_embedding) + '\n')
                        cnt += 1
                line = f.readline()
        embedding.append([0.0 for i in range(100)])
        embedding = np.array(embedding)
        pickle.dump(embedding, open(pickle_path, 'w'))
        of.close()

    def construct_symp_dict(self):
        """ construct symptom dict """
        return self.construct_dict(self.sym_path, self.sym_pickle, self.sym_dict, 0, False, ',')

    def construct_diag_dict(self):
        """ construct mention dict """
        return self.construct_dict(self.diag_path, self.diag_pickle, self.men_dict, 1, True)

    def read_data(self):
        """ process emr data and split into train and test  """
        of = open(self.res_path, 'w')
        with open(self.src_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                seg_list = line.split('@')
                if (len(seg_list)) != 3:
                    print "error! {} contains @ does't equal 2 ".format(line)
                symptoms, mentions, entities = seg_list
                symptoms = self.map2id(symptoms, self.sym_dict, ',')
                mentions = self.map2id(mentions, self.men_dict)
                entities = self.map2id(entities, self.en_dict)
                of.write('\t'.join(symptoms) + '@' + '\t'.join(mentions) + '@' + '\t'.join(entities) + '\n')
                line = f.readline()
        of.close()

    def map2id(self, m_list, dict_name, sep='\t'):
        """ transpose ids of symptom/mention to unique id and restore vec """
        new_list = []
        for m_ in m_list.split(sep):
            if m_ not in dict_name:
                new_list.append(len(dict_name))
                print '{} not in the dict, please check it!'.format(m_)
            else:
                new_list.append(dict_name[m_])
        return map(str, new_list)

if __name__ == '__main__':

    # load paramenters
    parser = argparse.ArgumentParser(description='Data Pre-Processing Parameters')
    parser.add_argument('--src-path', type=str, default='../../res/train_data_v3/labeled_emr2id_output.txt')
    parser.add_argument('--vec-path', type=str, default='../../res/train_data_v3/char2vec_id.vec')
    parser.add_argument('--res-path', type=str, default='../../res/train_data_v3/train_data.txt')
    parser.add_argument('--sym-vec-path', type=str, default='../../res/train_data_v3/sym_vec.vec')
    parser.add_argument('--diag-vec-path', type=str, default='../../res/train_data_v3/diag_vec.vec')
    parser.add_argument('--sym-pickle', type=str, default='../../res/train_data_v3/sym_vec.npy')
    parser.add_argument('--diag-pickle', type=str, default='../../res/train_data_v3/diag_vec.npy')

    args = parser.parse_args()

    dp = DataProcess(args.src_path, args.vec_path, args.res_path, args.sym_vec_path, args.diag_vec_path,
                     args.sym_pickle, args.diag_pickle)
    # load word embedding
    dp.read_word_embedding()
    print "load word embedding done! "

    # load symptom embedding
    dp.construct_symp_dict()
    print "load symptom embedding done !"

    # load diagnose embedding
    dp.construct_diag_dict()
    print "load diagnose embedding done! "

    # load emr data
    dp.read_data()

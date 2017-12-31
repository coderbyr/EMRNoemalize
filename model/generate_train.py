#!/usr/bin/env python
# coding: utf-8 

import os 
import re
import sys
import argparse
import pandas as pd 
import numpy as np 
import gensim as gs 
import cPickle as pickle
from gensim.models.keyedvectors import KeyedVectors
from scipy import spatial

class WordVector:
    """
    Load word dict and translate word embedding into wordId embedding
    input: original word2vec, word + embedding
    output: wordId + embedding
    """
    def __init__(self, word2vec_path, wordIdvec_path):
        self.word2vec_path = word2vec_path
        self.wordIdvec_path = wordIdvec_path
        self.word_dict = {}
        self.word_embed = {}

    def loadVocab(self):
        """ load word embedding and get vocab 
            save word dict and wordId embedding
        """
        of = open(self.wordIdvec_path, 'w')
        with open(self.word2vec_path, 'r') as f:
            line = f.readline()
            word_num, vec_dim = line.split()
            print 'word num is {}, vector length is {} .'.format(word_num, vec_dim)
            idx = 0
            line = f.readline()
            while line:
                line = line.strip()
                word, vec = line.split(' ', 1)
                self.word_dict[word.decode('utf8')] = idx
                self.word_embed[idx] = vec
                line = f.readline()
                of.write(str(idx) + '\t' + vec + '\n')
                idx += 1
        of.close()
        

class GenerateTrain:
    """
    load labeld_emr data which contains context, mentions and entities to generate training data
    """
    def __init__(self, word_dict, word_embed, labeled_emr, emr_output, emrId_output, entityId_output, 
                 entityVec_output, symId_output, symVec_output, mentionId_output, mentionVec_output, 
                 normal_name='#'):
        """ 
        Args:
            word_dict        : store word and word_idx
            word_embed       : store wordId and embedding 
            labeled_emr      : labeled emr data, which contains context, mentions, entities
            emr_output       : store filtered labeld emr data
            emrId_output     : translate word of emr data into ids
            entityId_output  : translate entity into ids
            symId_output     : translate symptom into id
            symVec_output    : save  symptom  embedding
            mentionId_output : translate mention into id 
            mentionVec_output: save  mention embedding
            normal_name      : load normal name 
        """
        self.word_dict         = word_dict
        self.word_embed        = word_embed
        self.labeled_emr       = labeled_emr
        self.emr_output        = emr_output
        self.emrId_output      = emrId_output
        self.entityId_output   = entityId_output
        self.entityVec_output  = entityVec_output
        self.symId_output      = symId_output
        self.symVec_output     = symVec_output
        self.mentionId_output  = mentionId_output
        self.mentionVec_output = mentionVec_output
        self.sym_dict          = {}
        self.mention_dict      = {}
        self.entity_dict       = {}

    def loadNormlName(self):
        """ load normal name file to dict """
        normal_name_list = pickle.load(open(self.normal_name, 'r'))
        for i in range(len(normal_name_list)):
            self.normal_name[normal_name_list[i].decode('utf8')] = i

    
    def filterUnlabelMention(self, mention_list, entity_list):
        """ filter unlabeled mention in original mention """
        if len(mention_list) != len(entity_list):
            return []
        new_mention_list = []
        new_entity_list = []
        for i in range(len(entity_list)):
            if (entity_list[i]) == '#':
                continue
            new_mention_list.append(mention_list[i])
            new_entity_list.append(entity_list[i])
        return [new_mention_list, new_entity_list]


    def getStrEmbedding(self, word_id_list, dim=100):
        embed = [0.0 for i in range(dim)]
        for i in word_id_list:
            if i not in self.word_embed:
                continue
            word_embed = self.word_embed[i]
            word_embed = [eval(i) for i in word_embed.split()]
            embed = map(sum, zip(embed, word_embed))
        return map(str, embed)


    def processList(self, s_dict, sym_list, cnt, of1, of2):
        """ process symptom list """
        res = []
        for s in sym_list:
            s = s.decode('utf8')
            word_list = list(s)
            word_id_list = [self.word_dict[i] if i in self.word_dict else len(self.word_dict) for i in word_list]
            if s not in s_dict:
                s_dict[s] = cnt
                sym_embed = self.getStrEmbedding(word_id_list)
                of1.write(str(cnt) + '\t' + s.encode('utf8') + '\t' + ' '.join(map(str, word_id_list)) + '\n')
                of2.write(str(cnt) +  '\t' + ' '.join(sym_embed) + '\n')
                cnt += 1  
            res.append(s_dict[s])
        return map(str, res), cnt 


    def readEMRData(self, thre=0.5):
        """ construct symptom dict, mention dict, entity_dict and translate """
        of  = open(self.symId_output, 'w')
        of2 = open(self.symVec_output, 'w')

        of3 = open(self.mentionId_output, 'w')
        of4 = open(self.mentionVec_output, 'w')

        of5 = open(self.entityId_output, 'w')
        of6 = open(self.entityVec_output, 'w')

        of7 = open(self.emr_output, 'w')
        of8 = open(self.emrId_output, 'w')

        with open(self.labeled_emr, 'r') as f:
            line = f.readline()
            cnt1, cnt2, cnt3 = 0, 0, 0
            while line:
                line = line.strip()
                seg_list = line.split('@')
                if len(seg_list) != 3:
                    line = f.readline()
                    print "error! each record should contains context, mentions and entities !"
                    continue
                context, mentions, entities = seg_list
                sym_list = context.split(',')
                mention_list = mentions.split('\t')
                entity_list = entities.split('\t')

                null_val_cnt = sum(i=='#' for i in entity_list)
                if null_val_cnt * 1.0 / len(entity_list) >= thre:
                    line = f.readline()
                    print "skip lines which contains too many null values !"
                    continue

                res = self.filterUnlabelMention(mention_list, entity_list)
                if not res:
                    line = f.readline()
                    print "error! mentions number should equal to entities !"
                    continue
                else:
                    mention_list, entity_list = res

                symId_list, cnt1     = self.processList(self.sym_dict, sym_list, cnt1, of, of2)
                mentionId_list, cnt2 = self.processList(self.mention_dict, mention_list, cnt2, of3, of4)
                entityId_list, cnt3  = self.processList(self.entity_dict, entity_list, cnt3, of5, of6)

                of7.write(context + '@' + '\t'.join(mention_list) + '@' + '\t'.join(entity_list) + '\n')
                of8.write('\t'.join(symId_list) + '\1' + '\t'.join(mentionId_list) + '\1' + '\t'.join(entityId_list) + '\n')
                line = f.readline()

        of.close()
        of2.close()
        of3.close()
        of4.close()
        of5.close()
        of6.close()
        of7.close()
        of8.close()
 

    def constructNPY(self, embed_path, output_npy_path):
        """ construct embedding into numpy """
        embed = []
        with open(embed_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.strip()
                word_id, word_vec = line.split('\t')
                word_vec = [eval(i) for i in word_vec.split()]
                embed.append(word_vec)
                line = f.readline()

        embed = np.array(embed)
        pickle.dump(embed, open(output_npy_path, 'w'))
    
    def calculateSimlarity(self, mention_path, entity_path, men_entity_sim_path):
        """ calculate cosine distance as similarity of mention and entity """
        sim = []
        mention_npy = pickle.load(open(mention_path, 'r'))
        entity_npy  = pickle.load(open(entity_path, 'r'))
        for i in range(len(mention_npy)):
            cos_sim = []
            for j in range(len(entity_npy)):
                cos_dis = 1 - spatial.distance.cosine(mention_npy[i], entity_npy[j])
                cos_sim.append(cos_dis)
            sim.append(cos_sim)
        sim = np.array(sim)
        pickle.dump(sim, open(men_entity_sim_path, 'w'))




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=' generate train data ')
    parser.add_argument('--word2vec-path', type=str, default='../res/char2vec/char2vec_100_1220.vec')
    parser.add_argument('--wordIdvec-path', type=str, default='../res/train_data_v3/char2vec_id.vec')
    parser.add_argument('--labeled-emr', type=str, default='../data/emr_process_data/head1000_content_mention_entity.txt')
    parser.add_argument('--emr_output', type=str, default='../res/train_data_v3/labeled_emr_output.txt')
    parser.add_argument('--emrId-output', type=str, default='../res/train_data_v3/labeled_emrId_output.txt')
    parser.add_argument('--entityId-output', type=str, default='../res/train_data_v3/entityId_output.txt')
    parser.add_argument('--entityVec-output', type=str, default='../res/train_data_v3/entityVec_output.txt')
    parser.add_argument('--symId-output', type=str, default='../res/train_data_v3/symId_output.txt')
    parser.add_argument('--symVec-output', type=str, default='../res/train_data_v3/symVec_output.txt')
    parser.add_argument('--mentionId-output', type=str, default='../res/train_data_v3/mentionId_output.txt')
    parser.add_argument('--mentionVec-output', type=str, default='../res/train_data_v3/mentionVec_output.txt')
    parser.add_argument('--simNpy-output', type=str, default='../res/train_data_v3/simNpy.npy')
    parser.add_argument('--symNpy-output', type=str, default='../res/train_data_v3/symNpy.npy')
    parser.add_argument('--mentionNpy-output', type=str, default='../res/train_data_v3/mentionNpy.npy')
    parser.add_argument('--entityNpy-output', type=str, default='../res/train_data_v3/entityNpy.npy')


    args = parser.parse_args()

    WV = WordVector(args.word2vec_path, args.wordIdvec_path)
    WV.loadVocab()
    print "load word dict done !"
    
    GT = GenerateTrain(WV.word_dict, WV.word_embed, args.labeled_emr, args.emr_output, args.emrId_output, 
                       args.entityId_output, args.entityVec_output, args.symId_output, args.symVec_output, 
                       args.mentionId_output, args.mentionVec_output)
    
    #GT.extractNormalName()
    #print "load entity name done !"

    GT.readEMRData()
    print "translate emr into emr2id done! "    

    GT.constructNPY(args.symVec_output, args.symNpy_output)
    print "construct sym embedding done !"

    GT.constructNPY(args.mentionVec_output, args.mentionNpy_output)
    print "construct mention embedding done !"

    GT.constructNPY(args.entityVec_output, args.entityNpy_output)
    print "construct mention embedding done !"

    GT.calculateSimlarity(args.mentionNpy_output, args.entityNpy_output, args.simNpy_output)
    print "calculate simlarity of mention embed and entity embed done !"

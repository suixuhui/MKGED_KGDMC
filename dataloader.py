import numpy as np
import torch

from torch.utils.data import Dataset
from options import args
import pickle
from utils import get_triple_neighbor, generate_anomalous_triples, dfs, get_triple_path, get_triple_neighbor_initial_wrong
import pathlib

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, model, mode, flag=False):
        self.len = len(triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.model = model
        self.mode = mode
        self.flag = flag
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx][0]
        triple_label = self.triples[idx][1]
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 10)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        positive_sample = torch.LongTensor(positive_sample)

        if self.flag:
            negative_sample = np.concatenate(negative_sample_list)
            negative_sample = torch.from_numpy(negative_sample)

            with torch.no_grad():
                positive_sample_extend = torch.unsqueeze(positive_sample, dim=0).cuda(args.gpu)
                negative_sample_extend = torch.unsqueeze(negative_sample, dim=0).cuda(args.gpu)
                negative_score = self.model.get_structure_score((positive_sample_extend, negative_sample_extend),
                                                                self.mode)
                negative_score = torch.squeeze(negative_score, dim=0)
                top_score, top_indices = torch.topk(negative_score, len(negative_score), largest=True, sorted=True)
                negative_sample = negative_sample[top_indices[-self.negative_sample_size:]]
        else:
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

        return positive_sample, negative_sample, self.mode, triple_label, idx


    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        triple_label = [_[3] for _ in data]
        idx = [_[4] for _ in data]
        return positive_sample, negative_sample, mode, triple_label, idx

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}
        for triple in triples:
            head, relation, tail = triple[0]
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples):
        self.len = len(triples)
        self.triples = triples

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx][0]
        triple_label = self.triples[idx][1]
        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, triple_label, idx

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        triple_label = [_[1] for _ in data]
        idx = [_[2] for _ in data]
        return positive_sample, triple_label, idx


class EntityErrorDetectionDataset(Dataset):
    def __init__(self, entities_label):
        self.len = len(entities_label)
        self.entities_label = entities_label

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        entity_id = idx
        entity_label = self.entities_label[idx]

        return entity_id, entity_label

    @staticmethod
    def collate_fn(data):
        entity_id = [_[0] for _ in data]
        entity_label = [_[1] for _ in data]
        return entity_id, entity_label


class TripleErrorDetectionDataset(Dataset):
    def __init__(self, triples):
        self.len = len(triples)
        self.triples = triples
        self.only_triples = [self.triples[i][0] for i in range(self.len)]
        self.neg_triples = generate_anomalous_triples(self.only_triples, args.num_entity, args.num_relation)
        h2t_fname = args.FILE_DIR + "/h2t"
        t2h_fname = args.FILE_DIR + "/t2h"
        A_fname = args.FILE_DIR + "/A_all"
        path_fname = args.FILE_DIR + "/path_triple"
        graph_fname = args.FILE_DIR + "/graph"
        entity_start_fname = args.FILE_DIR + "/entity_start"
        entity_end_fname = args.FILE_DIR + "/entity_end"
        initial_score_fname = args.FILE_DIR + "/initial_scores"
        self.h2t = pickle.load(open(h2t_fname, 'rb'))
        self.t2h = pickle.load(open(t2h_fname, 'rb'))
        self.A = pickle.load(open(A_fname, 'rb'))
        self.path = pickle.load(open(path_fname, 'rb'))
        self.graph = pickle.load(open(graph_fname, 'rb'))
        self.entity_start = pickle.load(open(entity_start_fname, 'rb'))
        self.entity_end = pickle.load(open(entity_end_fname, 'rb'))
        self.initial_wrong_triples = None
        if pathlib.Path(initial_score_fname).is_file():
            self.initial_scores = torch.load(initial_score_fname)
            top_score, top_indices = torch.topk(self.initial_scores, 40000, largest=True, sorted=True)
            ratio = 0.05
            self.initial_wrong_triples = [self.only_triples[i] for i in top_indices[:int(self.len * ratio)]]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx][0]
        triple_label = self.triples[idx][1]
        if self.initial_wrong_triples is None:
            hrt_neighbor = get_triple_neighbor(positive_sample[0], positive_sample[1], positive_sample[2], self.h2t,
                                               self.t2h, self.A, args.neighbor_number)
        else:
            hrt_neighbor = get_triple_neighbor_initial_wrong(positive_sample[0], positive_sample[1], positive_sample[2], self.h2t,
                                               self.t2h, self.A, args.neighbor_number, self.initial_wrong_triples)
        pos_path = self.path[idx]
        negative_sample = self.neg_triples[idx]
        head = negative_sample[0]
        tail = negative_sample[2]
        if head not in self.entity_start or tail not in self.entity_end:
            neg_path = [[negative_sample]]
        else:
            neg_paths_dots = []
            for start in self.entity_start[head]:
                visited = set()
                path = []
                dfs(self.graph, start, visited, path, self.entity_end[tail], neg_paths_dots)
            neg_path = []
            for p in neg_paths_dots:
                neg_path.append([self.only_triples[i] for i in p])
            neg_path.append([negative_sample])
        if self.initial_wrong_triples is None:
            negative_hrt_neighbor = get_triple_neighbor(negative_sample[0], negative_sample[1], negative_sample[2],
                                                        self.h2t, self.t2h, self.A, args.neighbor_number)
        else:
            negative_hrt_neighbor = get_triple_neighbor_initial_wrong(negative_sample[0], negative_sample[1], negative_sample[2],
                                                        self.h2t, self.t2h, self.A, args.neighbor_number,
                                                        self.initial_wrong_triples)
        hrt_neighbor = torch.LongTensor([hrt_neighbor, negative_hrt_neighbor])
        pos_final_path = get_triple_path(pos_path, args.path_number)
        neg_final_path = get_triple_path(neg_path, args.path_number)
        path = [pos_final_path, neg_final_path]
        return hrt_neighbor, triple_label, idx, path

    @staticmethod
    def collate_fn(data):
        hrt_neighbor = torch.stack([_[0] for _ in data], dim=0)
        triple_label = [_[1] for _ in data]
        idx = [_[2] for _ in data]
        path = [_[3] for _ in data]
        return hrt_neighbor, triple_label, idx, path


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
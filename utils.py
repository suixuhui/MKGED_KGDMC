import torch
import numpy as np
import random
import copy


def toarray(x):
    return torch.from_numpy(np.array(list(x)).astype(np.int32))


def generate_anomalous_triples(triples, num_entity, num_relation):
    neg_triples = []
    for head, rel, tail in triples:
        head_or_tail = random.randint(0, 2)
        if head_or_tail == 0:
            new_head = random.randint(0, num_entity - 1)
            new_relation = rel
            new_tail = tail
        elif head_or_tail == 1:
            new_head = head
            new_relation = random.randint(0, num_relation - 1)
            new_tail = tail
        else:
            new_head = head
            new_relation = rel
            new_tail = random.randint(0, num_entity - 1)
        anomaly = (new_head, new_relation, new_tail)
        while anomaly in triples:
            if head_or_tail == 0:
                new_head = random.randint(0, num_entity - 1)
                new_relation = rel
                new_tail = tail
            elif head_or_tail == 1:
                new_head = head
                new_relation = random.randint(0, num_relation - 1)
                new_tail = tail
            else:
                new_head = head
                new_relation = rel
                new_tail = random.randint(0, num_entity - 1)
            anomaly = (new_head, new_relation, new_tail)
        neg_triples.append(anomaly)
    return neg_triples


def get_neighbor_id(ent, h2t, t2h, A):

    hrt1 = []
    hrt2 = []
    if ent in h2t.keys():
        tails = list(h2t[ent])
        for i in tails:
            for rel in A[(ent, i)]:
                hrt1.append((ent, rel, i))

    if ent in t2h.keys():
        heads = list(t2h[ent])
        for i in heads:
            for rel in A[(i, ent)]:
                hrt2.append((i, rel, ent))

    hrt = hrt1 + hrt2

    return hrt


def get_neighbor_id_initial_wrong(ent, h2t, t2h, A, initial_wrong_triples):
    hrt1 = []
    hrt2 = []

    if ent in h2t.keys():
        tails = list(h2t[ent])
        for i in tails:
            for rel in A[(ent, i)]:
                if (ent, rel, i) not in initial_wrong_triples:
                    hrt1.append((ent, rel, i))

    if ent in t2h.keys():
        heads = list(t2h[ent])
        for i in heads:
            for rel in A[(i, ent)]:
                if (i, rel, ent) not in initial_wrong_triples:
                    hrt2.append((i, rel, ent))

    hrt = hrt1 + hrt2

    return hrt


def get_triple_neighbor(h, r, t, h2t, t2h, A, num_neighbor):

    head_neighbor = get_neighbor_id(h, h2t, t2h, A)
    tail_neighbor = get_neighbor_id(t, h2t, t2h, A)
    if len(head_neighbor) > num_neighbor:
        hh_neighbors = random.sample(head_neighbor, k=num_neighbor)
    elif num_neighbor > len(head_neighbor):
        if len(head_neighbor) > 0:
            hh_neighbors = random.choices(head_neighbor, k=num_neighbor)
        else:
            temp = [(h, r, t)]
            hh_neighbors = random.choices(temp, k=num_neighbor)
    else:
        hh_neighbors = head_neighbor

    if len(tail_neighbor) > num_neighbor:
        tt_neighbors = random.sample(tail_neighbor, k=num_neighbor)
    elif num_neighbor > len(tail_neighbor):
        if len(tail_neighbor) > 0:
            tt_neighbors = random.choices(tail_neighbor, k=num_neighbor)
        else:
            temp = [(h, r, t)]
            tt_neighbors = random.choices(temp, k=num_neighbor)
    else:
        tt_neighbors = tail_neighbor

    hrt_neighbor = [(h, r, t)] + hh_neighbors + [(h, r, t)] + tt_neighbors

    return hrt_neighbor


def get_triple_neighbor_initial_wrong(h, r, t, h2t, t2h, A, num_neighbor, initial_wrong_triples):

    head_neighbor = get_neighbor_id_initial_wrong(h, h2t, t2h, A, initial_wrong_triples)
    tail_neighbor = get_neighbor_id_initial_wrong(t, h2t, t2h, A, initial_wrong_triples)
    if len(head_neighbor) > num_neighbor:
        hh_neighbors = random.sample(head_neighbor, k=num_neighbor)
    elif num_neighbor > len(head_neighbor):
        if len(head_neighbor) > 0:
            hh_neighbors = random.choices(head_neighbor, k=num_neighbor)
        else:
            temp = [(h, r, t)]
            hh_neighbors = random.choices(temp, k=num_neighbor)
    else:
        hh_neighbors = head_neighbor

    if len(tail_neighbor) > num_neighbor:
        tt_neighbors = random.sample(tail_neighbor, k=num_neighbor)
    elif num_neighbor > len(tail_neighbor):
        if len(tail_neighbor) > 0:
            tt_neighbors = random.choices(tail_neighbor, k=num_neighbor)
        else:
            temp = [(h, r, t)]
            tt_neighbors = random.choices(temp, k=num_neighbor)
    else:
        tt_neighbors = tail_neighbor

    hrt_neighbor = [(h, r, t)] + hh_neighbors + [(h, r, t)] + tt_neighbors

    return hrt_neighbor


def dfs(graph, start, visited, path, ends, paths):
    visited.add(start)
    path.append(start)

    if start in ends:
        end_path = copy.deepcopy(path)
        paths.append(end_path)
        visited.remove(start)
        path.pop()
        return

    if start not in graph or len(path) > 4:
        visited.remove(start)
        path.pop()
        return

    for node in graph[start]:
        if node not in visited:
            dfs(graph, node, visited, path, ends, paths)

    visited.remove(start)
    path.pop()


def get_triple_path(path, number):
    if len(path) > number:
        final_path = random.sample(path, k=number)
    elif number > len(path):
        final_path = random.choices(path, k=number)
    else:
        final_path = path
    return final_path

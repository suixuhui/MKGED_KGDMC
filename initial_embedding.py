import copy
import json
import gensim
from nltk.tokenize import word_tokenize
import pathlib
import numpy as np
import pickle
import os
from transformers import BertTokenizer, ViTFeatureExtractor
from PIL import Image
from random import shuffle
import torch


# Get embedding of words from gensim word2vec model
def getEmbeddings(model, phr_list, embed_dims):
    embed_list = []
    all_num, oov_num, oov_rate = 0, 0, 0
    for phr in phr_list:
        if phr in model.vocab:
            embed_list.append(model.word_vec(phr))
            all_num += 1
        else:
            vec = np.zeros(embed_dims, np.float32)
            wrds = word_tokenize(phr)
            for wrd in wrds:
                all_num += 1
                if wrd in model.vocab:
                    vec += model.word_vec(wrd)
                else:
                    vec += np.random.randn(embed_dims)
                    oov_num += 1
            if len(wrds) == 0:
                embed_list.append(vec / 10000)
            else:
                embed_list.append(vec / len(wrds))
    oov_rate = oov_num / all_num
    print('oov rate:', oov_rate, 'oov num:', oov_num, 'all num:', all_num)
    return np.array(embed_list)


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


def get_initial_embedding(dataset, entity_support, relation_support, triples_list):
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
    special_tokens_dict = {
        "additional_special_tokens": [
            "[unused0]",
        ],
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    triples_id_fname = './datasets/' + dataset + '/file' + '/triples_id'
    if not pathlib.Path(triples_id_fname).is_file():
        new_triples = []
        for mode, triples in triples_list.items():
            for trp in triples:
                sub, rel, obj = trp
                sub_id, rel_id, obj_id = entity_support[sub]['token_id'], relation_support[rel]['token_id'], entity_support[obj]['token_id']
                triple_label = 0
                if mode == 'anomaly':
                    triple_label = 1
                new_trp = ((sub_id, rel_id, obj_id), triple_label)
                new_triples.append(new_trp)
        shuffle(new_triples)
        pickle.dump(new_triples, open(triples_id_fname, 'wb'))
    else:
        new_triples = pickle.load(open(triples_id_fname, 'rb'))

    h2t_fname = './datasets/' + dataset + '/file' + '/h2t'
    t2h_fname = './datasets/' + dataset + '/file' + '/t2h'
    if not pathlib.Path(h2t_fname).is_file() or not pathlib.Path(t2h_fname).is_file():
        h2t = {}
        t2h = {}
        for triple in new_triples:
            sub, rel, obj = triple[0]
            if sub not in h2t:
                h2t[sub] = [obj]
            else:
                if obj not in h2t[sub]:
                    h2t[sub].append(obj)
            if obj not in t2h:
                t2h[obj] = [sub]
            else:
                if sub not in t2h[obj]:
                    t2h[obj].append(sub)
        pickle.dump(h2t, open(h2t_fname, 'wb'))
        pickle.dump(t2h, open(t2h_fname, 'wb'))

    A_fname = './datasets/' + dataset + '/file' + '/A_all'
    if not pathlib.Path(A_fname).is_file():
        A = {}
        for triple in new_triples:
            sub, rel, obj = triple[0]
            # A[(sub, obj)] = rel
            if (sub, obj) not in A:
                A[(sub, obj)] = [rel]
            else:
                if rel not in A[(sub, obj)]:
                    A[(sub, obj)].append(rel)
        pickle.dump(A, open(A_fname, 'wb'))

    path_fname = './datasets/' + dataset + '/file' + '/path'
    if not pathlib.Path(path_fname).is_file():
        all_triples = [triple[0] for triple in new_triples]
        graph = {}
        starts = {}
        ends = {}
        for i in range(len(all_triples)):
            triple = all_triples[i]
            head, rel, tail = triple
            for j in range(len(all_triples)):
                headj, relj, tailj = all_triples[j]
                if tail == headj:
                    if i in graph:
                        graph[i].append(j)
                    else:
                        graph[i] = [j]
                if head == headj:
                    if i in starts:
                        starts[i].append(j)
                    else:
                        starts[i] = [j]
                if tail == tailj:
                    if i in ends:
                        ends[i].append(j)
                    else:
                        ends[i] = [j]
        all_path = {}
        for i in range(len(all_triples)):
            paths = []
            for start in starts[i]:
                visited = set()
                path = []
                dfs(graph, start, visited, path, ends[i], paths)
            print(paths)
            all_path[i] = paths
        pickle.dump(all_path, open(path_fname, 'wb'))
    else:
        all_path = pickle.load(open(path_fname, 'rb'))

    path_triple_fname = './datasets/' + dataset + '/file' + '/path_triple'
    if not pathlib.Path(path_triple_fname).is_file():
        all_triples = [triple[0] for triple in new_triples]
        path_triple = []
        for i in range(len(all_triples)):
            p_triple = []
            for p in all_path[i]:
                p_triple.append([all_triples[j] for j in p])
            path_triple.append(p_triple)
        pickle.dump(path_triple, open(path_triple_fname, 'wb'))

    graph_fname = './datasets/' + dataset + '/file' + '/graph'
    entity_start_fname = './datasets/' + dataset + '/file' + '/entity_start'
    entity_end_fname = './datasets/' + dataset + '/file' + '/entity_end'
    if not pathlib.Path(graph_fname).is_file() or not pathlib.Path(entity_start_fname).is_file() or not pathlib.Path(entity_end_fname).is_file():
        all_triples = [triple[0] for triple in new_triples]
        graph = {}
        starts = {}
        ends = {}
        for i in range(len(all_triples)):
            triple = all_triples[i]
            head, rel, tail = triple
            for j in range(len(all_triples)):
                headj, relj, tailj = all_triples[j]
                if tail == headj:
                    if i in graph:
                        graph[i].append(j)
                    else:
                        graph[i] = [j]
            if head in starts:
                starts[head].append(i)
            else:
                starts[head] = [i]
            if tail in ends:
                ends[tail].append(i)
            else:
                ends[tail] = [i]
        pickle.dump(graph, open(graph_fname, 'wb'))
        pickle.dump(starts, open(entity_start_fname, 'wb'))
        pickle.dump(ends, open(entity_end_fname, 'wb'))

    E_tokens_id_fname = './datasets/' + dataset + '/file' + '/E_tokens_id'
    if not pathlib.Path(E_tokens_id_fname).is_file():
        max_length = 128
        E_tokens_id = []
        for entity_id, entity_su in entity_support.items():
            e_name, e_desc = entity_su['name'], entity_su['desc']
            wps_name = tokenizer.tokenize(e_name)
            wps_desc = tokenizer.tokenize(e_desc)
            entity_tokens = [tokenizer.cls_token] + wps_name + ["[unused0]"] + wps_desc
            max_len = max_length - 1
            if len(entity_tokens) > max_len:
                entity_tokens = entity_tokens[:max_len]
            entity_tokens = entity_tokens + [tokenizer.sep_token]
            tokens_id = tokenizer.convert_tokens_to_ids(entity_tokens)
            E_tokens_id.append(tokens_id)
        pickle.dump(E_tokens_id, open(E_tokens_id_fname, 'wb'))
    else:
        E_tokens_id = pickle.load(open(E_tokens_id_fname, 'rb'))

    R_tokens_id_fname = './datasets/' + dataset + '/file' + '/R_tokens_id'
    if not pathlib.Path(R_tokens_id_fname).is_file():
        max_length = 10
        R_tokens_id = []
        for relation_id, relation_su in relation_support.items():
            r_name = relation_su['name']
            wps_name = tokenizer.tokenize(r_name)
            relation_tokens = [tokenizer.cls_token] + wps_name
            max_len = max_length - 1
            if len(relation_tokens) > max_len:
                relation_tokens = relation_tokens[:max_len]
            relation_tokens = relation_tokens + [tokenizer.sep_token]
            tokens_id = tokenizer.convert_tokens_to_ids(relation_tokens)
            R_tokens_id.append(tokens_id)
        pickle.dump(R_tokens_id, open(R_tokens_id_fname, 'wb'))
    else:
        R_tokens_id = pickle.load(open(R_tokens_id_fname, 'rb'))


    E_img_pixel_values_fname = './datasets/' + dataset + '/file' + '/E_img_pixel_values'
    entities_label_fname = './datasets/' + dataset + '/file' + '/entities_label'
    if not pathlib.Path(E_img_pixel_values_fname).is_file() or not pathlib.Path(entities_label_fname).is_file():
        data_path = './datasets/' + dataset
        image_change_samples_path = os.path.join(data_path, "img_change_samples")
        image_change_samples = pickle.load(open(image_change_samples_path, 'rb'))
        if dataset == "fb15k-237":
            img_data_path = './datasets/fb15k'
        else:
            img_data_path = './datasets/wnimgs'
        model_name = "vit-base-patch16-224-in21k"
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        E_img_pixel_values = []
        entities_label = []
        for entity_id, entity_su in entity_support.items():
            entity_label = 0
            change_entity_id = entity_id
            if entity_id in image_change_samples:
                change_entity_id = image_change_samples[entity_id]
                entity_label = 1
            if dataset == "fb15k-237":
                new_entity_id = change_entity_id[1:].replace("/", ".")
            else:
                new_entity_id = "n" + change_entity_id
            entity_image_file = os.path.join(img_data_path, new_entity_id)
            if len(os.listdir(entity_image_file)) != 1:
                pixel_values = torch.zeros((3, 224, 224))
            else:
                img_name = os.path.join(entity_image_file, os.listdir(entity_image_file)[0])
                try:
                    entity_img = Image.open(img_name).resize((224, 224), Image.Resampling.LANCZOS)
                    pixel_values = feature_extractor(entity_img, return_tensors='pt')["pixel_values"].squeeze()
                except:
                    pixel_values = torch.zeros((3, 224, 224))
            E_img_pixel_values.append(pixel_values)
            entities_label.append(entity_label)
        pickle.dump(E_img_pixel_values, open(E_img_pixel_values_fname, 'wb'))
        pickle.dump(entities_label, open(entities_label_fname, 'wb'))
    else:
        E_img_pixel_values = pickle.load(open(E_img_pixel_values_fname, 'rb'))


def read_support(dataset):
    data_path = './datasets/' + dataset
    entity_path = os.path.join(data_path, 'support', 'entity.json')
    entities = json.load(open(entity_path, 'r', encoding='utf-8'))
    for idx, e in enumerate(entities):  # 14541
        name = entities[e]['name']
        desc = entities[e]['desc']
        entities[e] = {
            'token_id': idx,  # used for filtering
            'desc': desc,  # entity description, which improve the performance significantly
            'name': name,  # meaningless for the model, but can be used to print texts for debugging
        }

    relation_path = os.path.join(data_path, 'support', 'relation.json')
    relations = json.load(open(relation_path, 'r', encoding='utf-8'))
    for idx, r in enumerate(relations):  # 237
        name = relations[r]['name']
        relations[r] = {
            'token_id': idx,
            'name': name,  # raw name of relations, we do not need new tokens to replace raw names
        }

    return entities, relations


def read_lines(dataset):
    data_path = './datasets/' + dataset
    data_paths = {
        'train': os.path.join(data_path, 'train.txt'),
        'dev': os.path.join(data_path, 'dev.txt'),
        'test': os.path.join(data_path, 'test.txt'),
        'anomaly': os.path.join(data_path, 'anomaly_triples.txt')
    }

    lines = dict()
    for mode in data_paths:
        data_path = data_paths[mode]
        raw_data = list()

        # 1. read triplets from files
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = str(line).strip().split('\t')
                raw_data.append((h, r, t))

        lines[mode] = raw_data

    return lines


dataset = "wn18rr"
entity_support, relation_support = read_support(dataset)
triples_list = read_lines(dataset)
# print(entity_support)
get_initial_embedding(dataset, entity_support, relation_support, triples_list)






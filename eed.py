import numpy as np
import torch

from options import args
import random
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam
import pickle
from models import MEED
from dataloader import *
from tqdm import tqdm

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    text_embedding_fname = args.FILE_DIR + '/text_entity_embedding_final'
    image_embedding_fname = args.FILE_DIR + '/image_entity_embedding_final'
    text_embedding = torch.load(text_embedding_fname)
    image_embedding = torch.load(image_embedding_fname)

    entities_label_fname = args.FILE_DIR + '/entities_label'
    entities_label = pickle.load(open(entities_label_fname, 'rb'))

    num_entities = len(entities_label)
    total_num_anomalies = args.num_entity_anomalies

    train_dataloader = DataLoader(
        EntityErrorDetectionDataset(entities_label),
        batch_size=args.vae_batch_size,
        shuffle=True,
        collate_fn=EntityErrorDetectionDataset.collate_fn
    )

    test_dataloader = DataLoader(
        EntityErrorDetectionDataset(entities_label),
        batch_size=args.vae_batch_size,
        shuffle=False,
        collate_fn=EntityErrorDetectionDataset.collate_fn
    )

    model = MEED(args, text_embedding, image_embedding)
    model.cuda(args.gpu)

    optimizer = Adam(
        model.parameters(),
        lr=args.vae_lr
    )

    model.train()

    soft_labels = None

    for epoch in range(args.vae_n_epochs):

        train_data_iter = iter(train_dataloader)
        num_iter = len(train_dataloader)

        for step in tqdm(range(num_iter)):
            entity_ids, entity_labels = next(train_data_iter)
            entity_ids = torch.LongTensor(entity_ids)
            entity_ids = entity_ids.cuda(args.gpu)
            if epoch < args.vae_threshold:
                loss, entity_score = model(entity_ids, soft_labels=None)
            else:
                loss, triple_score = model(entity_ids, soft_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch: ", epoch)
        all_entity_score = {}
        soft_labels = {}
        with torch.no_grad():
            test_iterator = iter(test_dataloader)
            for step in tqdm(range(len(test_iterator))):
                entity_ids, entity_labels = next(test_iterator)
                entity_ids = torch.LongTensor(entity_ids)
                entity_ids = entity_ids.cuda(args.gpu)
                loss, entity_score = model(entity_ids, soft_labels=None)
                entity_ids = entity_ids.cpu()

                for i in range(entity_ids.size(0)):
                    if entity_ids[i] not in all_entity_score:
                        all_entity_score[entity_ids[i].item()] = [entity_score[i].cpu()]
                    else:
                        all_entity_score[entity_ids[i].item()].append(entity_score[i].cpu())

        entity_scores = []
        for i in range(num_entities):
            if i not in all_entity_score:
                score = torch.tensor(0)
            else:
                score = sum(all_entity_score[i]) / len(all_entity_score[i])
            entity_scores.append(score)
            soft_labels[i] = score

        if args.dataset == "fb15k-237":
            entity_max_top_k = 2000
        else:
            entity_max_top_k = 4000

        entity_scores = np.array(entity_scores)
        entity_scores = torch.from_numpy(entity_scores)

        max_score = torch.max(entity_scores)
        min_score = torch.min(entity_scores)

        for code, label in soft_labels.items():
            soft_labels[code] = 1 - (soft_labels[code] - min_score) / (max_score - min_score)

        top_entity_score, top_entity_indices = torch.topk(entity_scores, len(entity_scores), largest=True, sorted=True)
        top_entity_labels = [entities_label[top_entity_indices[iii]] for iii in range(len(top_entity_indices))]

        entity_anomaly_discovered = []
        for i in range(entity_max_top_k):
            if i == 0:
                entity_anomaly_discovered.append(top_entity_labels[i])
            else:
                entity_anomaly_discovered.append(entity_anomaly_discovered[i - 1] + top_entity_labels[i])

        ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
        for i in range(len(ratios)):
            num_k = int(ratios[i] * num_entities)

            if num_k > len(entity_anomaly_discovered):
                break

            recall = entity_anomaly_discovered[num_k - 1] * 1.0 / args.num_entity_anomalies
            precision = entity_anomaly_discovered[num_k - 1] * 1.0 / num_k

            print(ratios[i], " entity precision: ", precision)
            print(ratios[i], " entity recall: ", recall)

        torch.save(entity_scores, args.FILE_DIR + "/entity_scores")

        sys.stdout.flush()


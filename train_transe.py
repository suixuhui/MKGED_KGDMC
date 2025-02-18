
import random
import sys
from torch.utils.data import DataLoader
from torch.optim import Adam
from models import TransE
from dataloader import *
import math
from utils import toarray
from tqdm import tqdm

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    triples_id_fname = args.FILE_DIR + '/triples_id'
    triples_id = pickle.load(open(triples_id_fname, 'rb'))

    nentity = args.num_entity
    nrelation = args.num_relation
    num_triples = args.num_triples
    total_num_anomalies = args.num_anomalies

    model = TransE(args, nentity, nrelation)

    train_dataloader_head = DataLoader(
        TrainDataset(triples_id, nentity, nrelation, args.transe_negative_number, model, 'head-batch'),
        batch_size=args.transe_batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    train_dataloader_tail = DataLoader(
        TrainDataset(triples_id, nentity, nrelation, args.transe_negative_number, model, 'tail-batch'),
        batch_size=args.transe_batch_size,
        shuffle=True,
        collate_fn=TrainDataset.collate_fn
    )

    train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

    test_dataloader = DataLoader(
        TestDataset(triples_id),
        batch_size=args.transe_batch_size,
        shuffle=False,
        collate_fn=TestDataset.collate_fn
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.transe_lr
    )

    num_iterations = math.ceil(len(triples_id) / args.transe_batch_size)

    model.train()

    soft_labels = None

    for epoch in range(args.transe_n_epochs):
        model.cuda(args.gpu)
        entity_norm = torch.norm(model.entity_embedding.weight.data, dim=1, keepdim=True)
        model.entity_embedding.weight.data = model.entity_embedding.weight.data / entity_norm

        for step in tqdm(range(num_iterations * 2)):
            positive_sample, negative_sample, mode, triple_label, idx = next(train_iterator)
            positive_sample = positive_sample.cuda(args.gpu)
            negative_sample = negative_sample.cuda(args.gpu)
            if epoch < args.transe_threshold:
                loss, triple_score = model(positive_sample, idx, negative_sample, mode, soft_labels=None)
            else:
                loss, triple_score = model(positive_sample, idx, negative_sample, mode, soft_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch: ", epoch)
        all_triple_score = []
        all_triple_label = []
        soft_labels = {}
        with torch.no_grad():
            test_iterator = iter(test_dataloader)
            for step in tqdm(range(len(test_iterator))):
                positive_sample, triple_label, idx = next(test_iterator)
                positive_sample = positive_sample.cuda(args.gpu)
                triple_score = model(positive_sample, idx, negative_sample=None, mode="test", soft_labels=None)
                triple_score = triple_score.cpu()
                for id, score in zip(idx, triple_score):
                    soft_labels[id] = score
                all_triple_score += triple_score
                all_triple_label += triple_label

        if args.dataset == "fb15k-237":
            max_top_k = 40000
            entity_max_top_k = 2000
            entity_total_num_anomalies = 727
        else:
            max_top_k = 10000
            entity_max_top_k = 4000
            entity_total_num_anomalies = 2047

        all_triple_score = np.array(all_triple_score)
        all_triple_score = torch.from_numpy(all_triple_score)

        max_score = torch.max(all_triple_score)
        min_score = torch.min(all_triple_score)

        for code, label in soft_labels.items():
            soft_labels[code] = 1 - (soft_labels[code] - min_score) / (max_score - min_score)

        top_score, top_indices = torch.topk(all_triple_score, len(all_triple_score), largest=True, sorted=True)
        top_labels = toarray([all_triple_label[top_indices[iii]] for iii in range(len(top_indices))])

        anomaly_discovered = []
        for i in range(max_top_k):
            if i == 0:
                anomaly_discovered.append(top_labels[i])
            else:
                anomaly_discovered.append(anomaly_discovered[i - 1] + top_labels[i])

        ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
        for i in range(len(ratios)):
            num_k = int(ratios[i] * num_triples)

            if num_k > len(anomaly_discovered):
                break

            recall = anomaly_discovered[num_k - 1] * 1.0 / total_num_anomalies
            precision = anomaly_discovered[num_k - 1] * 1.0 / num_k

            print(ratios[i], " triple precision: ", precision.item())
            print(ratios[i], " triple recall: ", recall.item())


        R_embedding = model.relation_embedding.cpu()
        torch.save(R_embedding, args.FILE_DIR + "/relation_embedding")

        sys.stdout.flush()


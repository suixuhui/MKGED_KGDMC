from options import args
import random
from torch.utils.data import DataLoader
from dataloader import *
import pickle
from tqdm import tqdm
from models import InitialScore
from utils import toarray


if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    text_embedding_fname = args.FILE_DIR + '/text_entity_embedding_final'
    image_embedding_fname = args.FILE_DIR + '/image_entity_embedding_final'
    relation_embedding_fname = args.FILE_DIR + '/relation_embedding'
    triples_id_fname = args.FILE_DIR + '/triples_id'
    entity_score_fname = args.FILE_DIR + '/entity_scores'
    text_embedding = torch.load(text_embedding_fname)
    image_embedding = torch.load(image_embedding_fname)
    relation_embedding = torch.load(relation_embedding_fname)
    triples_id = pickle.load(open(triples_id_fname, 'rb'))
    entity_scores = torch.load(entity_score_fname)

    num_triples = args.num_triples
    total_num_anomalies = args.num_anomalies

    dataloader = DataLoader(
        TripleErrorDetectionDataset(triples_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TripleErrorDetectionDataset.collate_fn
    )

    model = InitialScore(args, text_embedding, image_embedding, relation_embedding, entity_scores)
    model.cuda(args.gpu)

    all_triple_score = []
    all_triple_label = []
    with torch.no_grad():
        test_iterator = iter(dataloader)
        for step in tqdm(range(len(test_iterator))):
            hrt_neighbor, triple_label, idx, path = next(test_iterator)
            hrt_neighbor = hrt_neighbor.cuda(args.gpu)
            triple_score = model(hrt_neighbor)
            all_triple_score += triple_score.cpu()
            all_triple_label += triple_label

    all_triple_score = np.array(all_triple_score)
    all_triple_score = torch.from_numpy(all_triple_score)

    top_score, top_indices = torch.topk(all_triple_score, len(all_triple_score), largest=True, sorted=True)
    top_labels = toarray([all_triple_label[top_indices[iii]] for iii in range(len(top_indices))])

    if args.dataset == "fb15k-237":
        max_top_k = 40000
    else:
        max_top_k = 10000

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

    torch.save(all_triple_score, args.FILE_DIR + "/initial_scores")
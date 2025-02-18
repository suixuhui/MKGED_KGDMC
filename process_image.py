import os

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import random
import pickle


def read_support():
    entity_path = "./datasets/wn18rr/support/entity.json"
    entities = json.load(open(entity_path, 'r', encoding='utf-8'))
    for idx, e in enumerate(entities):  # 14541
        new_name = f'[E_{idx}]'
        raw_name = entities[e]['name']
        desc = entities[e]['desc']
        entities[e] = {
            'token_id': idx,  # used for filtering
            'name': new_name,  # new token to be added in tokenizer because raw name may consist many tokens
            'desc': desc,  # entity description, which improve the performance significantly
            'raw_name': raw_name,  # meaningless for the model, but can be used to print texts for debugging
        }

    return entities


def retain_only_one_image():
    dir_path = "./datasets/wnimgs"
    image_processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32").feature_extractor
    clip = CLIPModel.from_pretrained("clip-vit-base-patch32").cuda(4)
    for file_name in os.listdir(dir_path):
        file_dir_name = os.path.join(dir_path, file_name)
        file_image_embeds = None
        image_names = []
        if len(os.listdir(file_dir_name)) == 1:
            continue
        for file in os.listdir(file_dir_name):
            img_name = os.path.join(file_dir_name, file)
            if os.path.isdir(img_name):
                for name in os.listdir(img_name):
                    name = os.path.join(img_name, name)
                    os.remove(name)
                os.rmdir(img_name)
                continue
            try:
                img = Image.open(img_name).resize((224, 224), Image.Resampling.LANCZOS)
                pixel_values = image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
            except:
                os.remove(img_name)
                continue
            pixel_values = torch.unsqueeze(pixel_values, dim=0).cuda(4)
            image_embeds = clip.get_image_features(pixel_values)
            # file_image_embeds.append(pixel_values)
            if file_image_embeds is None:
                file_image_embeds = image_embeds
            else:
                file_image_embeds = torch.cat((file_image_embeds, image_embeds))
            image_names.append(img_name)
        if file_image_embeds is None:
            continue
        distances = torch.norm(file_image_embeds.unsqueeze(1) - file_image_embeds.unsqueeze(0), dim=2)
        distances = distances.sum(1)
        retain_index = torch.argmin(distances, dim=-1)
        retain_name = image_names[retain_index]
        for image in image_names:
            if image != retain_name:
                os.remove(image)


def gen_anomaly():
    entities = read_support()
    tokenizer = CLIPProcessor.from_pretrained("clip-vit-base-patch32").tokenizer
    dir_path = "./datasets/wnimgs"
    image_processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32").feature_extractor
    clip = CLIPModel.from_pretrained("clip-vit-base-patch32").cuda(4)
    img_entity_ids = []
    all_image_embeds = None
    for entity_id, entity_values in entities.items():
        # new_entity_id = entity_id[1:].replace("/", ".")
        new_entity_id = "n" + entity_id
        file_dir_name = os.path.join(dir_path, new_entity_id)
        if len(os.listdir(file_dir_name)) != 1:
            continue
        img_name = os.path.join(file_dir_name, os.listdir(file_dir_name)[0])
        try:
            img = Image.open(img_name).resize((224, 224), Image.Resampling.LANCZOS)
            pixel_values = image_processor(img, return_tensors='pt')['pixel_values'].squeeze()
        except:
            continue
        pixel_values = torch.unsqueeze(pixel_values, dim=0).cuda(4)
        with torch.no_grad():
            image_embeds = clip.get_image_features(pixel_values)
        img_entity_ids.append(entity_id)
        if all_image_embeds is None:
            all_image_embeds = image_embeds
        else:
            all_image_embeds = torch.cat((all_image_embeds, image_embeds))
    all_image_embeds = all_image_embeds.cpu()

    num_anomalies = int(0.05 * len(entities))
    sample_num = num_anomalies // 3
    idx = random.sample(range(0, len(entities) - 1), int(sample_num))
    selected_samples1 = [list(entities.keys())[idx[i]] for i in range(len(idx))]
    change_samples1 = []
    for sample in selected_samples1:
        new_sample_id = random.randint(0, len(entities) - 1)
        while sample == list(entities.keys())[new_sample_id]:
            new_sample_id = random.randint(0, len(entities) - 1)
        change_samples1.append((sample, list(entities.keys())[new_sample_id]))

    idx = random.sample(range(0, len(entities) - 1), int(sample_num * 1.1))
    selected_samples2_init = [list(entities.keys())[idx[i]] for i in range(len(idx))]
    selected_samples2 = []
    for sample in selected_samples2_init:
        if sample not in selected_samples1:
            selected_samples2.append(sample)
    selected_samples2 = selected_samples2[:int(sample_num)]
    change_samples2 = []
    for sample in selected_samples2:
        entity_values = entities[sample]
        entity, description = entity_values["raw_name"], entity_values["desc"]
        input_text = entity + ' [SEP] ' + description  # concat entity and sentence
        input_dict = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=77, truncation=True).to(4)
        with torch.no_grad():
            text_embeds = clip.get_text_features(**input_dict).cpu()
        euclidean_distances = []

        for img_embed in all_image_embeds:
            diff = text_embeds - img_embed
            squared_diff = diff ** 2
            distance = torch.sqrt(squared_diff.sum())
            euclidean_distances.append(distance.item())
        euclidean_distances = torch.tensor(euclidean_distances)
        index = torch.argmin(euclidean_distances, dim=-1)
        change_samples2.append((sample, img_entity_ids[index]))

    idx = random.sample(range(0, len(img_entity_ids) - 1), int(sample_num * 1.2))
    selected_samples3_init = [img_entity_ids[idx[i]] for i in range(len(idx))]
    selected_samples3 = []
    selected_idxs = []
    for idx, sample in zip(idx, selected_samples3_init):
        if sample not in selected_samples1 and sample not in selected_samples2:
            selected_samples3.append(sample)
            selected_idxs.append(idx)
    selected_samples3 = selected_samples3[:int(sample_num)]
    selected_idxs = selected_idxs[:num_anomalies - 2 * int(sample_num)]
    change_samples3 = []
    for idx, sample in zip(selected_idxs, selected_samples3):

        euclidean_distances = []
        selected_embed = all_image_embeds[idx]
        for img_embed in all_image_embeds:
            diff = selected_embed - img_embed
            squared_diff = diff ** 2
            distance = torch.sqrt(squared_diff.sum())
            euclidean_distances.append(distance.item())
        euclidean_distances = torch.tensor(euclidean_distances)
        index = torch.argsort(euclidean_distances, dim=-1)[1]
        change_samples3.append((sample, img_entity_ids[index]))

    image_change_samples = change_samples1 + change_samples2 + change_samples3
    all_image_change_samples = {}
    for samples in image_change_samples:
        all_image_change_samples[samples[0]] = samples[1]
    pickle.dump(all_image_change_samples, open('./datasets/wn18rr/img_change_samples', 'wb'), -1)

gen_anomaly()
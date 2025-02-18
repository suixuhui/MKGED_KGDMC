import time

import torch.nn as nn
from transformers import BertModel, ViTModel, BertTokenizer
import torch.nn.functional as F
import torch
from options import args
import numpy as np
import math


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, z_dim, output_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class TransE(nn.Module):

    def __init__(self, args, nentity, nrelation):
        super(TransE, self).__init__()
        self.name = args.model
        self.gpu = args.gpu

        self.dim = 768
        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.empty(nentity, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)), freeze=False)
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(nrelation, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)),
            freeze=False)

        # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm

        entity_norm = torch.norm(self.entity_embedding.weight.data, dim=1, keepdim=True)
        self.entity_embedding.weight.data = self.entity_embedding.weight.data / entity_norm

    def forward(self, positive_sample, idx, negative_sample, mode, soft_labels):

        positive_score = self.get_structure_score(positive_sample)
        if mode == "test":
            positive_score = torch.squeeze(positive_score)
            return positive_score

        negative_score = self.get_structure_score((positive_sample, negative_sample), mode)
        positive_score_repeat = positive_score.repeat(1, args.transe_negative_number)
        gamma = torch.full((1, args.transe_negative_number), float(args.gama)).cuda(self.gpu)
        loss = self.hinge_loss(positive_score_repeat, negative_score, gamma)
        if soft_labels is not None:
            batch_soft_labels = torch.tensor([soft_labels[id] for id in idx]).cuda(self.gpu)
            loss = batch_soft_labels * loss
        loss = loss.sum().requires_grad_()
        positive_score = torch.squeeze(positive_score)

        return loss, positive_score

    def get_structure_score(self, sample, mode='single'):

        if mode == 'single':
            head = self.entity_embedding(sample[:, 0]).unsqueeze(1)
            relation = self.relation_embedding(sample[:, 1]).unsqueeze(1)
            tail = self.entity_embedding(sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = self.entity_embedding(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = self.relation_embedding(tail_part[:, 1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = self.entity_embedding(head_part[:, 0]).unsqueeze(1)
            relation = self.relation_embedding(head_part[:, 1]).unsqueeze(1)
            tail = self.entity_embedding(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = head + relation - tail

        score = torch.norm(score, p=2, dim=2)

        return score

    def hinge_loss(self, positive_score, negative_score, gamma):
        err = positive_score - negative_score + gamma
        max_err = err.clamp(0)
        return max_err


class PretrainBERT(nn.Module):

    def __init__(self, args, tokens_id, relation_embedding):
        super(PretrainBERT, self).__init__()
        self.name = args.model
        self.gpu = args.gpu

        self.tokens_id = tokens_id
        self.relation_embedding = relation_embedding

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, do_lower_case=False)
        special_tokens_dict = {
            "additional_special_tokens": [
                "[unused0]",
            ],
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)

        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert.resize_token_embeddings(len(self.tokenizer))

    def forward(self, positive_sample, idx, negative_sample, mode, soft_labels):

        positive_score = self.get_structure_score(positive_sample)
        if mode == "test":
            positive_score = torch.squeeze(positive_score)
            if positive_sample.size(0) == 1:
                positive_score = torch.unsqueeze(positive_score, dim=0)
            return positive_score

        negative_score = self.get_structure_score((positive_sample, negative_sample), mode)
        positive_score_repeat = positive_score.repeat(1, args.bert_negative_number)
        gamma = torch.full((1, args.bert_negative_number), float(args.gama)).cuda(self.gpu)
        loss = self.hinge_loss(positive_score_repeat, negative_score, gamma)
        loss = torch.sum(loss, dim=-1)
        if soft_labels is not None:
            batch_soft_labels = torch.tensor([soft_labels[id] for id in idx]).cuda(self.gpu)
            loss = batch_soft_labels * loss
        loss = loss.sum().requires_grad_()
        positive_score = torch.squeeze(positive_score)

        return loss, positive_score

    def get_structure_score(self, samples, mode='single'):

        if mode == 'single':

            head = None
            relation = None
            tail = None

            for sample in samples:
                head_input_id = self.tokens_id[sample[0]]
                tail_input_id = self.tokens_id[sample[2]]
                head_input_id, tail_input_id = torch.LongTensor(head_input_id).cuda(self.gpu), torch.LongTensor(
                    tail_input_id).cuda(self.gpu)
                head_input_id, tail_input_id = torch.unsqueeze(head_input_id, dim=0), torch.unsqueeze(tail_input_id,
                                                                                                      dim=0)
                head_output_bert = self.bert(head_input_id)
                tail_output_bert = self.bert(tail_input_id)
                head_output_embedding = head_output_bert.last_hidden_state[0][0]
                tail_output_embedding = tail_output_bert.last_hidden_state[0][0]

                relation_output_embedding = self.relation_embedding(sample[1])

                if head is None:
                    head = head_output_embedding
                else:
                    head = torch.cat((head, head_output_embedding))
                if relation is None:
                    relation = relation_output_embedding
                else:
                    relation = torch.cat((relation, relation_output_embedding))
                if tail is None:
                    tail = tail_output_embedding
                else:
                    tail = torch.cat((tail, tail_output_embedding))

            head = head.view(-1, 768)
            relation = relation.view(-1, 768)
            tail = tail.view(-1, 768)

            head = head.unsqueeze(1)
            relation = relation.unsqueeze(1)
            tail = tail.unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = samples
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = None
            relation = None
            tail = None

            for sample in tail_part:
                tail_input_id = self.tokens_id[sample[2]]
                tail_input_id = torch.LongTensor(tail_input_id).cuda(self.gpu)
                tail_input_id = torch.unsqueeze(tail_input_id, dim=0)
                tail_output_bert = self.bert(tail_input_id)
                tail_output_embedding = tail_output_bert.last_hidden_state[0][0]

                relation_output_embedding = self.relation_embedding(sample[1])

                if relation is None:
                    relation = relation_output_embedding
                else:
                    relation = torch.cat((relation, relation_output_embedding))
                if tail is None:
                    tail = tail_output_embedding
                else:
                    tail = torch.cat((tail, tail_output_embedding))

            relation = relation.view(-1, 768)
            tail = tail.view(-1, 768)

            relation = relation.unsqueeze(1)
            tail = tail.unsqueeze(1)

            for sample in head_part:
                head_embedding = None
                for id in sample:
                    head_input_id = self.tokens_id[id]
                    head_input_id = torch.LongTensor(head_input_id).cuda(self.gpu)
                    head_input_id = torch.unsqueeze(head_input_id, dim=0)
                    head_output_bert = self.bert(head_input_id)
                    head_output_embedding = head_output_bert.last_hidden_state[0][0]
                    if head_embedding is None:
                        head_embedding = head_output_embedding
                    else:
                        head_embedding = torch.cat((head_embedding, head_output_embedding))
                head_embedding = head_embedding.view(-1, 768)

                if head is None:
                    head = head_embedding
                else:
                    head = torch.cat((head, head_embedding))

            head = head.view(-1, negative_sample_size, 768)


        elif mode == 'tail-batch':
            head_part, tail_part = samples
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = None
            relation = None
            tail = None

            for sample in head_part:
                head_input_id = self.tokens_id[sample[0]]
                head_input_id = torch.LongTensor(head_input_id).cuda(self.gpu)
                head_input_id = torch.unsqueeze(head_input_id, dim=0)
                head_output_bert = self.bert(head_input_id)
                head_output_embedding = head_output_bert.last_hidden_state[0][0]

                relation_output_embedding = self.relation_embedding(sample[1])

                if head is None:
                    head = head_output_embedding
                else:
                    head = torch.cat((head, head_output_embedding))
                if relation is None:
                    relation = relation_output_embedding
                else:
                    relation = torch.cat((relation, relation_output_embedding))

            head = head.view(-1, 768)
            relation = relation.view(-1, 768)

            head = head.unsqueeze(1)
            relation = relation.unsqueeze(1)

            for sample in tail_part:
                tail_embedding = None
                for id in sample:
                    tail_input_id = self.tokens_id[id]
                    tail_input_id = torch.LongTensor(tail_input_id).cuda(self.gpu)
                    tail_input_id = torch.unsqueeze(tail_input_id, dim=0)
                    tail_output_bert = self.bert(tail_input_id)
                    tail_output_embedding = tail_output_bert.last_hidden_state[0][0]
                    if tail_embedding is None:
                        tail_embedding = tail_output_embedding
                    else:
                        tail_embedding = torch.cat((tail_embedding, tail_output_embedding))
                tail_embedding = tail_embedding.view(-1, 768)

                if tail is None:
                    tail = tail_embedding
                else:
                    tail = torch.cat((tail, tail_embedding))

            tail = tail.view(-1, negative_sample_size, 768)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = head + relation - tail

        score = torch.norm(score, p=2, dim=2)

        return score

    def hinge_loss(self, positive_score, negative_score, gamma):
        err = positive_score - negative_score + gamma
        max_err = err.clamp(0)
        return max_err


class PretrainVIT(nn.Module):

    def __init__(self, args, pixel_values, relation_embedding):
        super(PretrainVIT, self).__init__()
        self.name = args.model
        self.gpu = args.gpu

        self.pixel_values = pixel_values
        self.relation_embedding = relation_embedding.cuda(self.gpu)
        self.relation_embedding.weight.requires_grad = False

        self.vit = ViTModel.from_pretrained(args.vit_dir)

    def forward(self, positive_sample, idx, negative_sample, mode, soft_labels):

        positive_score = self.get_structure_score(positive_sample)
        if mode == "test":
            positive_score = torch.squeeze(positive_score)
            if positive_sample.size(0) == 1:
                positive_score = torch.unsqueeze(positive_score, dim=0)
            return positive_score

        negative_score = self.get_structure_score((positive_sample, negative_sample), mode)
        positive_score_repeat = positive_score.repeat(1, args.vit_negative_number)
        gamma = torch.full((1, args.vit_negative_number), float(args.gama)).cuda(self.gpu)
        loss = self.hinge_loss(positive_score_repeat, negative_score, gamma)
        loss = torch.sum(loss, dim=-1)
        if soft_labels is not None:
            batch_soft_labels = torch.tensor([soft_labels[id] for id in idx]).cuda(self.gpu)
            loss = batch_soft_labels * loss
        loss = loss.sum().requires_grad_()
        positive_score = torch.squeeze(positive_score)

        return loss, positive_score

    def get_structure_score(self, samples, mode='single'):

        if mode == 'single':

            head = None
            relation = None
            tail = None

            for sample in samples:
                head_pixel_value = self.pixel_values[sample[0]]
                tail_pixel_value = self.pixel_values[sample[2]]
                head_pixel_value, tail_pixel_value = head_pixel_value.cuda(self.gpu), tail_pixel_value.cuda(self.gpu)
                head_pixel_value, tail_pixel_value = torch.unsqueeze(head_pixel_value, dim=0), torch.unsqueeze(tail_pixel_value,dim=0)
                head_output_vit = self.vit(head_pixel_value)
                tail_output_vit = self.vit(tail_pixel_value)
                head_output_embedding = head_output_vit.last_hidden_state[0][0]
                tail_output_embedding = tail_output_vit.last_hidden_state[0][0]
                relation_output_embedding = self.relation_embedding(sample[1])

                if head is None:
                    head = head_output_embedding
                else:
                    head = torch.cat((head, head_output_embedding))
                if relation is None:
                    relation = relation_output_embedding
                else:
                    relation = torch.cat((relation, relation_output_embedding))
                if tail is None:
                    tail = tail_output_embedding
                else:
                    tail = torch.cat((tail, tail_output_embedding))

            head = head.view(-1, 768)
            relation = relation.view(-1, 768)
            tail = tail.view(-1, 768)

            head = head.unsqueeze(1)
            relation = relation.unsqueeze(1)
            tail = tail.unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = samples
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = None
            relation = None
            tail = None

            for sample in tail_part:
                tail_pixel_value = self.pixel_values[sample[2]]
                tail_pixel_value = tail_pixel_value.cuda(self.gpu)
                tail_pixel_value = torch.unsqueeze(tail_pixel_value, dim=0)
                tail_output_vit = self.vit(tail_pixel_value)
                tail_output_embedding = tail_output_vit.last_hidden_state[0][0]
                relation_output_embedding = self.relation_embedding(sample[1])

                if relation is None:
                    relation = relation_output_embedding
                else:
                    relation = torch.cat((relation, relation_output_embedding))
                if tail is None:
                    tail = tail_output_embedding
                else:
                    tail = torch.cat((tail, tail_output_embedding))

            relation = relation.view(-1, 768)
            tail = tail.view(-1, 768)

            relation = relation.unsqueeze(1)
            tail = tail.unsqueeze(1)

            for sample in head_part:
                head_embedding = None
                for id in sample:
                    head_pixel_value = self.pixel_values[id]
                    head_pixel_value = head_pixel_value.cuda(self.gpu)
                    head_pixel_value = torch.unsqueeze(head_pixel_value, dim=0)
                    head_output_vit = self.vit(head_pixel_value)
                    head_output_embedding = head_output_vit.last_hidden_state[0][0]
                    if head_embedding is None:
                        head_embedding = head_output_embedding
                    else:
                        head_embedding = torch.cat((head_embedding, head_output_embedding))
                head_embedding = head_embedding.view(-1, 768)

                if head is None:
                    head = head_embedding
                else:
                    head = torch.cat((head, head_embedding))

            head = head.view(-1, negative_sample_size, 768)


        elif mode == 'tail-batch':
            head_part, tail_part = samples
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = None
            relation = None
            tail = None

            for sample in head_part:
                head_pixel_value = self.pixel_values[sample[0]]
                head_pixel_value = head_pixel_value.cuda(self.gpu)
                head_pixel_value = torch.unsqueeze(head_pixel_value, dim=0)
                head_output_vit = self.vit(head_pixel_value)
                head_output_embedding = head_output_vit.last_hidden_state[0][0]
                relation_output_embedding = self.relation_embedding(sample[1])

                if head is None:
                    head = head_output_embedding
                else:
                    head = torch.cat((head, head_output_embedding))
                if relation is None:
                    relation = relation_output_embedding
                else:
                    relation = torch.cat((relation, relation_output_embedding))

            head = head.view(-1, 768)
            relation = relation.view(-1, 768)

            head = head.unsqueeze(1)
            relation = relation.unsqueeze(1)

            for sample in tail_part:
                tail_embedding = None
                for id in sample:
                    tail_pixel_value = self.pixel_values[id]
                    tail_pixel_value = tail_pixel_value.cuda(self.gpu)
                    tail_pixel_value = torch.unsqueeze(tail_pixel_value, dim=0)
                    tail_output_vit = self.vit(tail_pixel_value)
                    tail_output_embedding = tail_output_vit.last_hidden_state[0][0]
                    if tail_embedding is None:
                        tail_embedding = tail_output_embedding
                    else:
                        tail_embedding = torch.cat((tail_embedding, tail_output_embedding))
                tail_embedding = tail_embedding.view(-1, 768)

                if tail is None:
                    tail = tail_embedding
                else:
                    tail = torch.cat((tail, tail_embedding))

            tail = tail.view(-1, negative_sample_size, 768)

        else:
            raise ValueError('mode %s not supported' % mode)

        score = head + relation - tail

        score = torch.norm(score, p=2, dim=2)

        return score

    def hinge_loss(self, positive_score, negative_score, gamma):
        err = positive_score - negative_score + gamma
        max_err = err.clamp(0)
        return max_err


class MEED(nn.Module):

    def __init__(self, args, text_embedding, image_embedding):
        super(MEED, self).__init__()
        self.name = args.model
        self.gpu = args.gpu

        self.text_embedding = text_embedding.cuda(self.gpu)
        self.image_embedding = image_embedding.cuda(self.gpu)

        self.text_common_vae = VAE(768, args.hidden_dim, args.latent_dim, args.latent_dim * 2, 768)
        self.image_common_vae = VAE(768, args.hidden_dim, args.latent_dim, args.latent_dim * 2, 768)
        self.text_private_vae = VAE(768, args.hidden_dim, args.latent_dim, args.latent_dim * 2, 768)
        self.image_private_vae = VAE(768, args.hidden_dim, args.latent_dim, args.latent_dim * 2, 768)

        self.loss_KLD = lambda mu, sigma: -0.5 * torch.mean(1 + torch.log(sigma ** 2) - mu.pow(2) - sigma ** 2)

    def obtain_text_embedding(self, samples):
        text_embedding = None

        for sample in samples:
            embedding = self.text_embedding[sample]
            if text_embedding is None:
                text_embedding = embedding
            else:
                text_embedding = torch.cat((text_embedding, embedding))

        text_embedding = text_embedding.view(-1, 768)
        return text_embedding

    def obtain_image_embedding(self, samples):
        image_embedding = None
        for sample in samples:
            embedding = self.image_embedding[sample]
            if image_embedding is None:
                image_embedding = embedding
            else:
                image_embedding = torch.cat((image_embedding, embedding))
        image_embedding = image_embedding.view(-1, 768)
        return image_embedding

    def vae_encoder_process(self, x, vae):
        encoded = vae.encoder(x)
        mu, logvar = torch.chunk(encoded, 2, dim=1)
        z = vae.reparameterize(mu, logvar)
        return mu, logvar, z

    def vae_decoder_process(self, x, z, vae):
        decoded = vae.decoder(z)
        re = torch.norm(x - decoded, p=2, dim=1)
        return re

    def forward(self, entity_ids, soft_labels):

        text_embedding = self.obtain_text_embedding(entity_ids)
        image_embedding = self.obtain_image_embedding(entity_ids)

        text_private_mu, text_private_logvar, text_private_z = self.vae_encoder_process(text_embedding, self.text_private_vae)
        image_private_mu, image_private_logvar, image_private_z = self.vae_encoder_process(image_embedding, self.image_private_vae)
        text_common_mu, text_common_logvar, text_common_z = self.vae_encoder_process(text_embedding, self.text_common_vae)
        image_common_mu, image_common_logvar, image_common_z = self.vae_encoder_process(image_embedding, self.image_common_vae)

        distance_common = torch.sqrt(torch.sum((text_common_mu - image_common_mu) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(text_common_logvar.exp()) - torch.sqrt(image_common_logvar.exp())) ** 2, dim=1))
        distance_common = distance_common.mean()

        distance_text = torch.sqrt(torch.sum((text_common_mu - text_private_mu) ** 2, dim=1) + torch.sum(
                        (torch.sqrt(text_common_logvar.exp()) - torch.sqrt(text_private_logvar.exp())) ** 2, dim=1))
        distance_text = distance_text.mean()

        distance_image = torch.sqrt(torch.sum((image_private_mu - image_common_mu) ** 2, dim=1) + torch.sum(
                                  (torch.sqrt(image_private_logvar.exp()) - torch.sqrt(image_common_logvar.exp())) ** 2, dim=1))
        distance_image = distance_image.mean()

        distance_private = torch.sqrt(torch.sum((text_private_mu - image_private_mu) ** 2, dim=1) + \
                                      torch.sum((torch.sqrt(text_private_logvar.exp()) - torch.sqrt(image_private_logvar.exp())) ** 2, dim=1))
        distance_private = distance_private.mean()

        distance = distance_common - args.alpha * (distance_text + distance_image) - args.beta * distance_private

        text_private_norm, image_private_norm, text_common_norm, image_common_norm = self.loss_KLD(text_private_mu, text_private_logvar), self.loss_KLD(
            image_private_mu, image_private_logvar), self.loss_KLD(text_common_mu, text_common_logvar), self.loss_KLD(image_common_mu, image_common_logvar)

        text_z = torch.cat((text_private_z, text_common_z), dim=-1)
        image_z = torch.cat((image_private_z, image_common_z), dim=-1)
        image_cross_z = torch.cat((image_private_z, text_common_z), dim=-1)

        text_re = self.vae_decoder_process(text_embedding, text_z, self.text_private_vae)
        image_re = self.vae_decoder_process(image_embedding, image_z, self.image_private_vae)
        image_cross_re = self.vae_decoder_process(text_embedding, image_cross_z, self.image_common_vae)

        re = text_re + image_re + image_cross_re

        if soft_labels is not None:
            batch_soft_labels = torch.tensor([soft_labels[id.item()] for id in entity_ids]).cuda(self.gpu)
            re = batch_soft_labels * re

        loss_re = re.mean()
        loss_norm = text_private_norm + text_common_norm + image_private_norm + image_common_norm

        loss = loss_re + loss_norm + distance

        return loss, image_cross_re


class InitialScore(nn.Module):

    def __init__(self, args, text_embedding, image_embedding, relation_embedding, entity_scores):
        super(InitialScore, self).__init__()
        self.name = args.model
        self.gpu = args.gpu

        self.text_embedding = nn.Embedding.from_pretrained(text_embedding.cuda(self.gpu))
        self.image_embedding = nn.Embedding.from_pretrained(image_embedding.cuda(self.gpu))
        self.relation_embedding = relation_embedding.cuda(self.gpu)

        self.entity_scores = entity_scores.cuda(self.gpu)
        self.max_score = torch.max(self.entity_scores)
        self.min_score = torch.min(self.entity_scores)

        self.num_neighbor = args.neighbor_number

    def aggregate_embedding(self, samples):

        head_text_output_embedding = self.text_embedding(samples[:, 0])
        tail_text_output_embedding = self.text_embedding(samples[:, 2])
        head_image_output_embedding = self.image_embedding(samples[:, 0])
        tail_image_output_embedding = self.image_embedding(samples[:, 2])
        relation = self.relation_embedding(samples[:, 1])
        head_entity_score = self.entity_scores[samples[:, 0]]
        tail_entity_score = self.entity_scores[samples[:, 2]]

        head_radio = (head_entity_score - self.min_score.repeat(head_entity_score.size(0))) / (
                    self.max_score - self.min_score)
        tail_radio = (tail_entity_score - self.min_score.repeat(head_entity_score.size(0))) / (
                    self.max_score - self.min_score)
        head_radio = torch.unsqueeze(head_radio, dim=1).repeat(1, 768)
        tail_radio = torch.unsqueeze(tail_radio, dim=1).repeat(1, 768)

        head = (head_text_output_embedding + head_radio * head_image_output_embedding) / (torch.tensor(1.0).cuda(self.gpu).repeat(head_entity_score.size(0), 768) + head_radio)
        tail = (tail_text_output_embedding + tail_radio * tail_image_output_embedding) / (torch.tensor(1.0).cuda(self.gpu).repeat(head_entity_score.size(0), 768) + tail_radio)

        return head, relation, tail

    def forward(self, hrt_neighbor):

        batch_head = None
        batch_relation = None
        batch_tail = None
        for neighbors in hrt_neighbor:
            for neighbor in neighbors:
                heads, relations, tails = self.aggregate_embedding(neighbor)
                if batch_head is None:
                    batch_head = heads
                else:
                    batch_head = torch.cat((batch_head, heads))
                if batch_relation is None:
                    batch_relation = relations
                else:
                    batch_relation = torch.cat((batch_relation, relations))
                if batch_tail is None:
                    batch_tail = tails
                else:
                    batch_tail = torch.cat((batch_tail, tails))

        batch_head = batch_head.view(-1, 2, (self.num_neighbor + 1) * 2, 768)
        batch_relation = batch_relation.view(-1, 2, (self.num_neighbor + 1) * 2, 768)
        batch_tail = batch_tail.view(-1, 2, (self.num_neighbor + 1) * 2, 768)
        positive_head = batch_head[:, 0, 0, :]
        positive_relation = batch_relation[:, 0, 0, :]
        positive_tail = batch_tail[:, 0, 0, :]
        local_structure_pos_score = self.get_structure_score(positive_head, positive_relation, positive_tail)

        return local_structure_pos_score

    def get_structure_score(self, head, relation, tail):
        score = head + relation - tail
        score = torch.norm(score, p=2, dim=1)
        return score


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.05)

    def forward(self, inp):
        """
        inp: input_fea [Batch_size, N, in_features]
        """
        h = torch.matmul(inp, self.W)
        N = h.size()[1]
        B = h.size()[0]

        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)
        a_input = torch.cat((h, a), dim=2)
        e = self.leakyrelu(torch.matmul(a_input, self.a))

        attention = F.softmax(e, dim=1)  # [batch_size, N, 1]
        attention = attention - 0.005
        attention = (attention + abs(attention)) / 2.0
        attention = F.dropout(attention, self.dropout, training=self.training)
        attention = attention.view(B, 1, N)
        h_prime = torch.matmul(attention, h).squeeze(
            1)

        return h_prime


class MKGED(nn.Module):

    def __init__(self, args, text_embedding, image_embedding, relation_embedding, entity_scores):
        super(MKGED, self).__init__()
        self.name = args.model
        self.gpu = args.gpu

        self.text_embedding = nn.Embedding.from_pretrained(text_embedding.cuda(self.gpu))
        self.image_embedding = nn.Embedding.from_pretrained(image_embedding.cuda(self.gpu))
        self.relation_embedding = relation_embedding.cuda(self.gpu)

        self.entity_scores = entity_scores.cuda(self.gpu)
        self.max_score = torch.max(self.entity_scores)
        self.min_score = torch.min(self.entity_scores)

        self.num_layers = args.num_layers
        self.num_neighbor = args.neighbor_number
        self.lstm = nn.LSTM(768, 768, self.num_layers, batch_first=True, bidirectional=True)
        self.attention = GraphAttentionLayer(768 * 2 * 3, 768 * 2 * 3, dropout=args.dropout)

    def aggregate_embedding(self, samples):

        head_text_output_embedding = self.text_embedding(samples[:, 0])
        tail_text_output_embedding = self.text_embedding(samples[:, 2])

        head_image_output_embedding = self.image_embedding(samples[:, 0])
        tail_image_output_embedding = self.image_embedding(samples[:, 2])
        relation = self.relation_embedding(samples[:, 1])
        head_entity_score = self.entity_scores[samples[:, 0]]
        tail_entity_score = self.entity_scores[samples[:, 2]]

        head_radio = (self.max_score.repeat(head_entity_score.size(0)) - head_entity_score) / self.max_score
        tail_radio = (self.max_score.repeat(head_entity_score.size(0)) - tail_entity_score) / self.max_score

        head_radio = torch.unsqueeze(head_radio, dim=1).repeat(1, 768)
        tail_radio = torch.unsqueeze(tail_radio, dim=1).repeat(1, 768)

        head = (head_text_output_embedding + head_radio * head_image_output_embedding)
        tail = (tail_text_output_embedding + tail_radio * tail_image_output_embedding)

        return head, relation, tail


    def forward(self, hrt_neighbor, idx, all_path, soft_labels):


        batch_size = hrt_neighbor.size(0)
        batch_head = None
        batch_relation = None
        batch_tail = None
        for neighbors in hrt_neighbor:
            for neighbor in neighbors:
                heads, relations, tails = self.aggregate_embedding(neighbor)
                if batch_head is None:
                    batch_head = heads
                else:
                    batch_head = torch.cat((batch_head, heads))
                if batch_relation is None:
                    batch_relation = relations
                else:
                    batch_relation = torch.cat((batch_relation, relations))
                if batch_tail is None:
                    batch_tail = tails
                else:
                    batch_tail = torch.cat((batch_tail, tails))
        batch_head = batch_head.view(-1, 768)
        batch_relation = batch_relation.view(-1, 768)
        batch_tail = batch_tail.view(-1, 768)

        batch_triples_emb = torch.cat((batch_head, batch_relation), dim=-1)
        batch_triples_emb = torch.cat((batch_triples_emb, batch_tail), dim=-1)
        x = batch_triples_emb.view(-1, 3, 768)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), 768).cuda(self.gpu)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), 768).cuda(self.gpu)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(-1, 768 * 2 * 3)
        out = out.reshape(-1, self.num_neighbor + 1, 768 * 2 * 3)

        out_att = self.attention(out)
        out_att = out_att.reshape(batch_size, -1, 768 * 2 * 3)

        pos_z0 = out_att[:, 0, :]
        pos_z1 = out_att[:, 1, :]
        neg_z0 = out_att[:, 2, :]
        neg_z1 = out_att[:, 3, :]
        global_structure_pos_score = torch.norm(pos_z0 - pos_z1, p=2, dim=1)
        global_structure_neg_score = torch.norm(neg_z0 - neg_z1, p=2, dim=1)
        gamma = torch.tensor(args.gama).cuda(self.gpu)
        global_structure_loss = self.hinge_loss(global_structure_pos_score, global_structure_neg_score, gamma)

        batch_head = batch_head.view(-1, 2, (self.num_neighbor + 1) * 2, 768)
        batch_relation = batch_relation.view(-1, 2, (self.num_neighbor + 1) * 2, 768)
        batch_tail = batch_tail.view(-1, 2, (self.num_neighbor + 1) * 2, 768)
        positive_head = batch_head[:, 0, 0, :]
        positive_relation = batch_relation[:, 0, 0, :]
        positive_tail = batch_tail[:, 0, 0, :]
        negative_head = batch_head[:, 1, 0, :]
        negative_relation = batch_relation[:, 1, 0, :]
        negative_tail = batch_tail[:, 1, 0, :]
        local_structure_pos_score = self.get_structure_score(positive_head, positive_relation, positive_tail)
        local_structure_neg_score = self.get_structure_score(negative_head, negative_relation, negative_tail)
        gamma = torch.tensor(args.gama).cuda(self.gpu)
        local_structure_loss = self.hinge_loss(local_structure_pos_score, local_structure_neg_score, gamma)

        batch_scores = None
        for paths_triple in all_path:
            for paths in paths_triple:
                paths_embedding = None
                scores = None
                for path in paths:
                    path = torch.LongTensor(path).cuda(self.gpu)
                    heads, relations, tails = self.aggregate_embedding(path)
                    score1 = self.get_structure_score(heads, relations, tails)
                    score1 = score1.mean()
                    path_embedding = self.relation_embedding(path[:, 1])
                    path_embedding = torch.sum(path_embedding, dim=0)
                    score2 = self.get_structure_score(torch.unsqueeze(heads[0], dim=0), torch.unsqueeze(path_embedding, dim=0), torch.unsqueeze(tails[-1], dim=0))
                    score2 = torch.squeeze(score2)
                    score = torch.tensor([score1, score2]).cuda(self.gpu)
                    if paths_embedding is None:
                        paths_embedding = path_embedding
                    else:
                        paths_embedding = torch.cat((paths_embedding, path_embedding))
                    if scores is None:
                        scores = score
                    else:
                        scores = torch.cat((scores, score))
                scores = scores.view(-1, 2)
                score_0 = scores[:, 0]
                radio = nn.functional.softmax(score_0, dim=-1)
                scores = torch.mean(radio * scores[:, 1], dim=0)
                scores = torch.unsqueeze(scores, dim=0)
                if batch_scores is None:
                    batch_scores = scores
                else:
                    batch_scores = torch.cat((batch_scores, scores))
        batch_scores = batch_scores.view(-1, 2)
        global_path_pos_score = batch_scores[:, 0]
        global_path_neg_score = batch_scores[:, 1]
        gamma = torch.tensor(args.gama).cuda(self.gpu)
        global_path_loss = self.hinge_loss(global_path_pos_score, global_path_neg_score, gamma)

        loss = local_structure_loss + global_structure_loss + global_path_loss
        pos_score = local_structure_pos_score + args.lam * 10 * global_structure_pos_score + args.delta * 10 * global_path_pos_score

        if soft_labels is not None:
            batch_soft_labels = torch.tensor([soft_labels[id] for id in idx]).cuda(self.gpu)
            loss = batch_soft_labels * loss
        loss = loss.sum().requires_grad_()
        return loss, pos_score

    def get_structure_score(self, head, relation, tail):
        score = head + relation - tail
        score = torch.norm(score, p=2, dim=1)
        return score

    def hinge_loss(self, positive_score, negative_score, gamma):
        err = positive_score - negative_score + gamma
        max_err = err.clamp(0)
        return max_err
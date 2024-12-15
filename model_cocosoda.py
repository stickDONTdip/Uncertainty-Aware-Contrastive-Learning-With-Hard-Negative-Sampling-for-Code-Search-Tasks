# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RobertaHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size * 4)
        self.out_proj = nn.Linear(config.hidden_size * 4, config.hidden_size)
        # self.activation = nn.Tanh()

    def forward(self, features, attention_mask=None):
        last_hidden = features.last_hidden_state
        x1 = last_hidden[:, 0]  # take <s> token (equiv. to [CLS])
        x2 = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))

        hidden_states = features.hidden_states
        first_hidden = hidden_states[1]
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]

        x3 = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            -1).unsqueeze(-1)
        x4 = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(
            -1).unsqueeze(-1)
        x = torch.cat([x1, x2, x3, x4], dim=-1)

        x = self.dropout(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x2, x


class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size * 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size * 2, 2)

    def forward(self, a, b):
        x = torch.cat([a, b, a * b], dim=-1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config):
        super(Model, self).__init__()
        # 要传入config 初始化不确定学习头

        self.encoder_code = encoder
        self.mu_head_code = RobertaHead(config)
        self.logvar_head_code = RobertaHead(config)

        self.mu_head_nl = RobertaHead(config)
        self.logvar_head_nl = RobertaHead(config)

        self.classification_head = RobertaClassificationHead(config)

    def _reparameterize(self, mu, stdvar):
        epsilon = torch.randn_like(stdvar).to(stdvar.device)
        return mu + epsilon * stdvar

    def KL_loss(self, mu=None, var=None):
        variance_dul = var.pow(2)
        kl_loss = ((variance_dul + mu.pow(2) - torch.log(variance_dul) - 1) * 0.5).mean()

        t_var = torch.reciprocal(var + 1e-8)
        bs, l = var.size()
        varsum = t_var.sum(dim=1)
        u_loss = l / varsum
        u_loss = u_loss.mean()
        return kl_loss, u_loss

    def classification_loss(self, nl_vec, code_vec):
        nl = torch.cat([nl_vec, nl_vec], dim=0)
        code = torch.cat(
            [code_vec, code_vec[torch.randperm(code_vec.size(0))]],
            dim=0)
        scores = self.classification_head(nl, code)
        loss_fct = CrossEntropyLoss()
        label1 = torch.ones(code_vec.size(0)).long()
        label2 = torch.zeros(code_vec.size(0)).long()
        label = torch.cat([label1, label2], dim=0).to(scores.device)
        loss = loss_fct(scores, label)
        return loss

    def cencoder(self, inputs=None, modal=None):
        x1 = self.encoder_code(inputs, attention_mask=inputs.ne(1), output_hidden_states=True)
        if modal == 'neg':
            last_hidden = x1.last_hidden_state
            attention_mask = inputs.ne(1)
            x2 = ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            return x2
        x, mu = self.mu_head_code(x1, inputs.ne(1))
        _, logvar = self.logvar_head_code(x1, inputs.ne(1))
        stdvar = (logvar * 0.5).exp()
        return x, mu, stdvar

    def sort_candidates(self, a, a_mu, a_var):
        candidates_size = 60
        sharp = torch.tensor([]).to(a.device)

        for i in range(candidates_size):
            sharp = torch.cat((sharp, self._reparameterize(a_mu, a_var)), dim=0)

        sharp = torch.reshape(sharp, (candidates_size, -1, a.size(1)))
        sharp = torch.stack([i for i in sharp], dim=1)
        sharp = torch.nn.functional.normalize(sharp, p=2, dim=-1)

        scores = torch.einsum('ik,ijk->ij', a, sharp)
        sorted_scores, indices = torch.sort(scores, descending=True)

        sorted_candidates = torch.tensor([]).to(sharp.device)
        for i, index in zip(sharp, indices[:, 0:1]):
            sorted_candidates = torch.cat((sorted_candidates, i[index]), dim=0)
        return sorted_candidates

    def forward(self, code_inputs=None, neg_inputs=None, nl_inputs=None, labels=None):

        if code_inputs is not None:
            code, mu_code, stdvar_code = self.cencoder(code_inputs, modal='code')
            code = nn.functional.normalize(code, p=2, dim=-1)
            if labels is not 'train':
                return code

        if nl_inputs is not None:
            nl, mu_nl, stdvar_nl = self.cencoder(nl_inputs)
            nl = nn.functional.normalize(nl, p=2, dim=-1)
            if labels is not 'train':
                return nl

        if neg_inputs is not None:
            neg_code = self.cencoder(neg_inputs, modal='neg')
            neg_code = nn.functional.normalize(neg_code, p=2, dim=-1)
            return neg_code

        nl_candidate = self.sort_candidates(nl, mu_nl, stdvar_nl)
        code_candidate = self.sort_candidates(code, mu_code, stdvar_code)

        k1, u1 = self.KL_loss(mu=mu_nl, var=stdvar_nl)
        k2, u2 = self.KL_loss(mu=mu_code, var=stdvar_code)
        kl_loss = (k1 + k2) / 2
        uncertain = (u1 + u2) / 2
        loss = 0.01 * kl_loss
        return loss, uncertain, nl, code, nl_candidate, code_candidate

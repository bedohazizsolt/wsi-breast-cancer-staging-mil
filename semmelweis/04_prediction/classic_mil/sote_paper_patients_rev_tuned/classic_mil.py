"""
Copyright 2026 Zsolt Bedőházi, András M. Biricz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class Feature_attention(nn.Module):
    def __init__(self, emb_dim, hidden_dim, att_hidden_dim, dropout_fc, dropout_attn, n_classes):
        super(Feature_attention, self).__init__()
        self.L = emb_dim
        self.I = hidden_dim # 256
        self.D = att_hidden_dim # 128
        self.K = 1

        self.feature_extractor = nn.Sequential(
            nn.Linear(self.L, self.L), 
            nn.LeakyReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(self.L, self.I), 
            nn.LeakyReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(self.I, self.I), 
            nn.LeakyReLU(),
            nn.Dropout(dropout_fc),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.I, self.D),
            nn.Tanh(),
            nn.Dropout(dropout_attn),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.I*self.K, self.I),
            nn.LeakyReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(self.I, n_classes)
        )

    def forward(self, x):
        x = x.squeeze(0)

        x = self.feature_extractor(x)
        A = self.attention(x) 
        A = torch.transpose(A, 1, 0) 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, x)
        logits = self.classifier(M)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, A


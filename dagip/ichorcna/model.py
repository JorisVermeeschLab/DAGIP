# -*- coding: utf-8 -*-
#
#  model.py
#
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import numpy as np
import torch
import torch.nn.functional
import tqdm

from dagip.nipt.binning import ChromosomeBounds


class IchorCNA(torch.nn.Module):

    def __init__(self, n_in: int):
        super().__init__()
        self.n_hidden: int = 64
        self.layers = torch.nn.Sequential(
            torch.nn.Conv1d(n_in, self.n_hidden, (5,), padding=(2,)),
            torch.nn.Tanh(),
            torch.nn.Conv1d(self.n_hidden, self.n_hidden, (5,), padding=(2,)),
            torch.nn.Tanh(),
            torch.nn.Conv1d(self.n_hidden, self.n_hidden, (5,), padding=(2,)),
            torch.nn.Tanh(),
            torch.nn.Conv1d(self.n_hidden, self.n_hidden, (5,), padding=(2,)),
            torch.nn.Tanh(),
            torch.nn.Conv1d(self.n_hidden, self.n_hidden, (5,), padding=(2,)),
            torch.nn.Tanh(),
            torch.nn.Conv1d(self.n_hidden, 1, (5,), padding=(2,))
        )
        self.apply(IchorCNA.init_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        bounds = ChromosomeBounds.get_1mb()
        chrs = []
        for c in range(22):
            start, end = bounds[c], bounds[c + 1]
            chrs.append(self.layers.forward(X[:, :, start:end]))
        X = torch.cat(chrs, dim=2)
        X = torch.squeeze(X)
        assert len(X.size()) == 1
        return X

    def decode(self, x: torch.Tensor, r: torch.Tensor, side_info: torch.Tensor) -> torch.Tensor:
        assert len(x.size()) == 1
        assert len(side_info.size()) == 2
        pc = 0.01
        x = torch.clamp(torch.log((x + pc) / (r + pc)), 0, 1)
        X = torch.cat((x.unsqueeze(0), side_info.t()), dim=0).unsqueeze(0)
        return self.forward(X)

    def fit(self, X: np.ndarray, side_info: np.ndarray, log_ratios: np.ndarray):
        r = torch.FloatTensor(np.median(X, axis=0))
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(log_ratios)
        side_info = torch.FloatTensor(side_info)

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.train()
        for _ in tqdm.tqdm(range(200), desc='Training ichorCNA'):
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            total_loss = 0.
            for i in idx:
                optimizer.zero_grad()
                y = Y[i, :]
                y_hat = self.decode(X[i, :], r, side_info)
                mask = (y != 0)
                loss = torch.mean((y_hat[mask] - y[mask]) ** 2)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            total_loss /= len(X)

            print(f'Loss: {total_loss}')

        self.eval()

    def predict(self, X: np.ndarray, side_info: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        m = X.shape[1]
        y_hat = np.empty((n, m))
        r = torch.FloatTensor(np.median(X, axis=0))
        X = torch.FloatTensor(X)
        side_info = torch.FloatTensor(side_info)

        for i in range(n):
            y_hat[i, :] = self.decode(X[i, :], r, side_info).cpu().data.numpy()
        return y_hat

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath, _use_new_zipfile_serialization=False)

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath), strict=False)

    @staticmethod
    def init_weights(m):
        if type(m) in [torch.nn.Conv1d, torch.nn.Linear]:
            print("Initializing weights...", m.__class__.__name__)
            # t.nn.init.normal(m.weight, 0, 0.01)
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.1)
        elif isinstance(m, torch.nn.Embedding):
            print("Initializing weights...", m.__class__.__name__)
            torch.nn.init.xavier_uniform_(m.weight)

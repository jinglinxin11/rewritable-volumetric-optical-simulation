"""Offline training and validation-only selection, before material compilation."""

import copy
from dataclasses import asdict
import time

import numpy as np
import torch
import torch.nn.functional as F

from .artifacts import save_json
from .config import NetworkConfig
from .network import ReferenceCNN


@torch.no_grad()
def evaluate(model, inputs, labels, batch_size=1024):
    model.eval()
    predictions = torch.cat([model.forward_patches(chunk).argmax(1).cpu() for chunk in inputs.split(batch_size)])
    return dict(accuracy_pct=float((predictions == labels.cpu()).sum()) / len(labels) * 100,
                predictions=predictions)


def train_reference(data, seed, output, epochs=20, batch_size=512, learning_rate=.005):
    if epochs < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive epochs, batch size and finite learning rate are required')
    torch.manual_seed(seed)
    model = ReferenceCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    order = torch.Generator().manual_seed(seed + 3000)
    tx, ty = data.train
    vx, vy = data.validation
    best, selected, selected_epoch, history = -1., None, None, []
    started = time.time()
    for epoch in range(epochs + 1):
        model.train()
        loss_sum = 0.
        if epoch:
            for ids in torch.randperm(len(ty), generator=order).split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.forward_patches(tx[ids]), ty[ids])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                model.constrain_weights()
                loss_sum += float(loss.detach()) * len(ids)
        val = evaluate(model, vx, vy)['accuracy_pct']
        history.append(dict(epoch=epoch, validation_accuracy_pct=val,
                            training_loss=loss_sum / len(ty) if epoch else None))
        if val > best:
            best, selected_epoch, selected = val, epoch, copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or epoch == epochs:
            print(f'seed{seed} epoch{epoch}: validation={val:.2f}%', flush=True)
    model.load_state_dict(selected)
    record = dict(seed=seed, selected_epoch=selected_epoch, validation_accuracy_pct=best,
                  elapsed_seconds=time.time() - started)
    torch.save(dict(schema=1, network=asdict(NetworkConfig()), state_dict=selected, **record), output / 'reference.pt')
    save_json(output / 'history.json', history)
    save_json(output / 'selected.json', record)
    return model, record

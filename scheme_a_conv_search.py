"""Validation-only bounded convolution search with fresh-seed confirmation.

All trainable weight matrices remain differential binary material banks. This is
an independent-intensity-channel simulation, not an optical propagation model.
"""
import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import scheme_a_multilayer_head as previous

base = previous.base
ROOT = Path(__file__).resolve().parent
ZONES = previous.ZONES
SAVE = base.prior.common.save_json


@dataclass(frozen=True)
class Architecture:
    channels: int = 3
    kernel: int = 3
    stride: int = 1

    def __post_init__(self):
        if self.channels < 1 or self.kernel not in (3, 5) or self.stride not in (1, 2):
            raise ValueError('Positive channels; kernel 3/5; stride 1/2 required')

    @property
    def side(self):
        return (14-self.kernel)//self.stride+1

    @property
    def widths(self):
        return (self.channels*9, 32, 16, 10)

    @property
    def name(self):
        return f'c{self.channels}_k{self.kernel}_s{self.stride}'


GRID = tuple(Architecture(c,k,s) for c in (3,6,12) for k in (3,5) for s in (1,2))


def shapes(config, arch):
    result = {'conv': (config.layers,2,arch.channels,arch.kernel**2)}
    result.update({f'fc{i+1}': (config.layers,2,b,a)
                   for i,(a,b) in enumerate(zip(arch.widths,arch.widths[1:]))})
    return result


def resources(config, arch):
    bank_sites = {z:int(np.prod(s)) for z,s in shapes(config,arch).items()}
    dense_weights = sum(a*b for a,b in zip(arch.widths,arch.widths[1:]))
    return dict(material_sites=sum(bank_sites.values()), bank_sites=bank_sites,
                electronic_biases=arch.channels+sum(arch.widths[1:]),
                effective_weights=arch.channels*arch.kernel**2+dense_weights,
                trainable_parameters=sum(bank_sites.values())+arch.channels+sum(arch.widths[1:]),
                conv_windows=arch.side**2,
                conv_channels=2*arch.channels*arch.kernel**2,
                channel_readings_per_image=2*(arch.side**2*arch.channels*arch.kernel**2+dense_weights),
                sequential_banks=4)


def patches(images, arch):
    x = torch.from_numpy(images).float()[:,None]/255
    x = F.interpolate(x, size=(14,14), mode='area')
    return F.unfold(x, kernel_size=arch.kernel, stride=arch.stride).transpose(1,2).contiguous()


class Model(nn.Module):
    def __init__(self, config, arch, seed):
        super().__init__()
        self.config, self.arch = config, arch
        for z,offset in zip(ZONES,(1000,4000,5000,6000)):
            g = torch.Generator().manual_seed(seed+offset)
            setattr(self,z+'_theta',nn.Parameter(.5*torch.randn(shapes(config,arch)[z],generator=g)))
        self.conv_bias = nn.Parameter(torch.zeros(arch.channels))
        for i,(a,b) in enumerate(zip(arch.widths,arch.widths[1:])):
            g = torch.Generator().manual_seed(seed+7000+i)
            setattr(self,f'fc{i+1}_bias',nn.Parameter((2*torch.rand(b,generator=g)-1)/a**.5))

    bits = previous.Model.bits
    weights = previous.Model.weights
    forward = previous.Model.forward

    def logits_from_weights(self,x,w):
        a = self.arch
        z = x@w['conv'].T
        h = F.relu(z.transpose(1,2).reshape(-1,a.channels,a.side,a.side)
                   +self.conv_bias[None,:,None,None])
        # Output remains a 3x3 grid per channel. Non-divisible sizes use
        # overlapping adaptive-average bins, not fixed 4x4 pooling.
        h = F.adaptive_avg_pool2d(h,(3,3)).flatten(1)
        for i in range(1,4):
            h = base.material_classifier(h,w[f'fc{i}'],getattr(self,f'fc{i}_bias'))
            if i<3:
                h = F.relu(h)
        return h


class Device(nn.Module):
    def __init__(self,config,arch):
        super().__init__()
        self.config, self.arch = config, arch
        for z,s in shapes(config,arch).items():
            self.register_buffer(z+'_state',torch.zeros(s))
        self.initialized = False
        self.requests = self.erase_events = self.skipped_identical = 0
        self.write_ones = dict.fromkeys(ZONES,0)
        self.last_written = dict.fromkeys(ZONES,0)

    weights = previous.Device.weights

    @torch.no_grad()
    def program(self,target):
        if set(target)!=set(ZONES):
            raise ValueError('Atomic programming requires all four banks')
        for z,s in shapes(self.config,self.arch).items():
            if tuple(target[z].shape)!=s or not bool(((target[z]==0)|(target[z]==1)).all()):
                raise ValueError('Invalid material mask')
        self.requests += 1
        if self.initialized and all(torch.equal(target[z],getattr(self,z+'_state')) for z in ZONES):
            self.skipped_identical += 1
            self.last_written = dict.fromkeys(ZONES,0)
            return False
        copied = {z:v.detach().clone() for z,v in target.items()}
        for z in ZONES:
            getattr(self,z+'_state').zero_()
        self.erase_events += 1
        for z,v in copied.items():
            getattr(self,z+'_state').copy_(v)
            self.last_written[z] = int(v.sum())
            self.write_ones[z] += self.last_written[z]
        self.initialized = True
        return True


def train(config,arch,seed,args,folder,tx,ty,vx,vy):
    folder.mkdir(parents=True,exist_ok=True)
    name = f'{arch.name}_seed{seed}'
    completed = folder/f'{name}_result.json'
    if completed.exists():
        return json.loads(completed.read_text())
    model,device = Model(config,arch,seed),Device(config,arch)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    generator = torch.Generator().manual_seed(seed+3000)
    initial = model.bits()
    device.program(initial)
    history,events = [],[]
    best,checkpoint = -1.,None
    started = time.time()
    for epoch in range(args.epochs+1):
        total,count = 0.,0
        grads = dict.fromkeys(ZONES,0.)
        if epoch:
            model.train()
            for ids in torch.randperm(len(ty),generator=generator).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(previous.training_logits(model,device,tx[ids]),ty[ids])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                for z in ZONES:
                    grads[z] += float(getattr(model,z+'_theta').grad.norm())
                optimizer.step()
                changed = device.program(model.bits())
                count += 1
                total += float(loss.detach())*len(ids)
                events.append(dict(epoch=epoch,batch=count,rewritten=changed,
                                   erase_events=device.erase_events,written_ones=dict(device.last_written)))
        accuracy,_ = base.evaluate(model,vx,vy,device)
        if accuracy>best:
            best = accuracy
            checkpoint = dict(config=asdict(config),arch=asdict(arch),seed=seed,
                              selected_epoch=epoch,validation_accuracy=accuracy,
                              model_state=copy.deepcopy(model.state_dict()),bits=model.bits())
        bits = model.bits()
        history.append(dict(epoch=epoch,validation_accuracy=accuracy,
                            training_loss=total/len(ty) if epoch else None,
                            gradient_norm={z:g/max(1,count) for z,g in grads.items()},
                            changed_bits={z:int((bits[z]!=initial[z]).sum()) for z in ZONES},
                            erase_events=device.erase_events,write_ones=dict(device.write_ones)))
        if epoch%10==0 or epoch==args.epochs:
            print(f'{folder.name}/{name} epoch={epoch} val={accuracy:.4f}',flush=True)
    torch.save(checkpoint,folder/f'{name}.pt')
    SAVE(folder/f'{name}_history.json',history)
    SAVE(folder/f'{name}_events.json',events)
    result = dict(architecture=arch.name,arch=asdict(arch),seed=seed,selected_epoch=checkpoint['selected_epoch'],
                  validation_accuracy=best,elapsed_seconds=time.time()-started,
                  erase_events=device.erase_events,write_ones=dict(device.write_ones),
                  resources=resources(config,arch))
    SAVE(completed,result)
    return result


def load_selected(path):
    ck = torch.load(path,weights_only=True)
    config,arch = base.prior.Config(**ck['config']),Architecture(**ck['arch'])
    model = Model(config,arch,ck['seed'])
    model.load_state_dict(ck['model_state'])
    for z,b in model.bits().items():
        assert torch.equal(b,ck['bits'][z])
    device = Device(config,arch)
    device.program(ck['bits'])
    return ck,model,device


def mask_rows(bits,config,arch):
    for depth in range(config.layers):
        offset = 0
        for z in ZONES:
            value = bits[z][depth]
            for branch in range(2):
                for out in range(value.shape[1]):
                    for inp in range(value.shape[2]):
                        if z=='conv':
                            row,col = branch,arch.kernel**2*out+inp
                        else:
                            row,col = offset+branch*value.shape[1]+out,inp
                        yield dict(layer_index=depth,zone=z,branch=branch,output_index=out,
                                   input_index=inp,x_um=col*config.center_pitch_um,
                                   y_um=row*config.center_pitch_um,write_bit=int(value[branch,out,inp]))
            offset += 2 if z=='conv' else 2*value.shape[1]


def summarize(rows,config):
    results = []
    for arch in GRID:
        selected = [r for r in rows if r['architecture']==arch.name]
        if not selected:
            continue
        acc = [r['validation_accuracy'] for r in selected]
        results.append(dict(architecture=arch.name,arch=asdict(arch),
                            validation_mean=float(np.mean(acc)),validation_population_sd=float(np.std(acc)),
                            seed_values=acc,resources=resources(config,arch)))
    return sorted(results,key=lambda r:(-r['validation_mean'],r['resources']['material_sites'],r['architecture']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/scheme_a_conv_search_20260925')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--lr',type=float,default=.005)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if min(args.epochs,args.batch_size)<1 or args.lr<=0:
        parser.error('Positive budgets required')
    if args.output.exists() and not args.resume:
        parser.error('Output exists; use a new output or --resume')
    args.output.mkdir(parents=True,exist_ok=True)
    config = base.prior.Config()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    sources = ['scheme_a_conv_search.py','scheme_a_multilayer_head.py','scheme_a_material_head.py',
               'scheme_a_binary10.py','scheme_a_closed_loop.py','scheme_a_round2.py','scheme_a_mnist.py','reproduce.py']
    plan = dict(grid=[asdict(a) for a in GRID],search_seeds=[0,1,2],confirmation_seeds=[3,4,5],
                epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,config=asdict(config),
                source_hashes={s:base.prior.common.hash_file(ROOT/s) for s in sources},
                versions=dict(torch=str(torch.__version__),numpy=np.__version__),
                selection='Mean best validation accuracy over search seeds; ties favor fewer material sites then name. Winner locked before fresh-seed confirmation. No test-based selection.',
                protocol='Full 50k training, fixed 10k validation; earliest best validation checkpoint; same batch ordering per seed; final test only winner and baseline at seeds 3,4,5.',
                limitations=[
                    'Only these 12 architectures; not a global optimum or a learning-rate/epoch/head-width search.',
                    '14x14 input, padding 0, adaptive average pooling to 3x3 per channel; classifier widths 9C->32->16->10.',
                    'Adaptive pooling equals original 4x4 pooling for the 12x12 baseline; other sizes use possibly overlapping bins.',
                    'Ten binary depth sites; T=1 or 0.9; all weights material; electronic activations and biases; no residuals.',
                    'Zero mismatch/noise/crosstalk, ideal intensity encoding, no coherent propagation or ADC/DAC costs.',
                    'Pitch 50um only determines mask coordinates, not optical isolation; not an exposure recipe.',
                    'Accuracy versus resource counts, not measured latency, energy or hardware throughput.',
                    'More kernels also enlarge the first classifier layer; not a parameter-count-matched ablation.',
                    'New seeds reduce seed-specific selection effects but reuse validation images; no nested validation or blind new test cohort.',
                    'Existing MNIST test set has been used in previous project analyses.'])
    plan_path = args.output/'search_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:
            raise ValueError('Resume plan/source mismatch')
    else:
        SAVE(plan_path,plan)
        snap = args.output/'source_snapshot'
        snap.mkdir()
        for s in sources:
            shutil.copy2(ROOT/s,snap/s)
    started = time.time()
    status = dict(status='searching',started_unix=started)
    SAVE(args.output/'status.json',status)
    raw,hashes = base.prior.common.r2.base.fetch_data(ROOT/'data/MNIST/raw')
    SAVE(args.output/'dataset_hashes.json',hashes)
    with np.load(ROOT/'outputs/scheme_a_mnist/split_indices.npz') as splits:
        tr,va = splits['train'],splits['validation']
        assert len(tr)==50000 and len(va)==10000 and not set(tr)&set(va)
        np.savez_compressed(args.output/'split_indices.npz',**{k:splits[k] for k in splits.files})
    y = torch.from_numpy(raw['train-labels-idx1-ubyte.gz'].astype(np.int64))
    ty,vy = y[tr],y[va]
    rows = []
    for arch in GRID:
        x = patches(raw['train-images-idx3-ubyte.gz'],arch)
        tx,vx = x[tr],x[va]
        del x
        for seed in plan['search_seeds']:
            rows.append(train(config,arch,seed,args,args.output/'search',tx,ty,vx,vy))
            SAVE(args.output/'search_results.json',rows)
        del tx,vx
        ranking = summarize(rows,config)
        SAVE(args.output/'validation_ranking.json',ranking)
        print('SEARCH_PROGRESS',len(rows),'/36',arch.name,'mean_val',
              next(r['validation_mean'] for r in ranking if r['architecture']==arch.name),flush=True)
    winner = Architecture(**summarize(rows,config)[0]['arch'])
    locked = dict(winner=asdict(winner),name=winner.name,
                  criterion=plan['selection'],all_candidates_completed=len(rows),
                  ranking=summarize(rows,config))
    lock_path = args.output/'locked_selection_before_confirmation.json'
    if lock_path.exists():
        assert json.loads(lock_path.read_text())==locked
    else:
        SAVE(lock_path,locked)
    print('LOCKED_WINNER',winner.name,flush=True)
    status['status'] = 'confirming'
    SAVE(args.output/'status.json',status)
    confirm = []
    candidates = [Architecture()]
    if winner!=candidates[0]:
        candidates.append(winner)
    for arch in candidates:
        x = patches(raw['train-images-idx3-ubyte.gz'],arch)
        tx,vx = x[tr],x[va]
        del x
        for seed in plan['confirmation_seeds']:
            confirm.append(train(config,arch,seed,args,args.output/'confirmation',tx,ty,vx,vy))
        del tx,vx
    SAVE(args.output/'confirmation_selection_before_test.json',confirm)
    status['status'] = 'testing'
    SAVE(args.output/'status.json',status)
    yt = torch.from_numpy(raw['t10k-labels-idx1-ubyte.gz'].astype(np.int64))
    for arch in candidates:
        xt = patches(raw['t10k-images-idx3-ubyte.gz'],arch)
        for row in [r for r in confirm if r['architecture']==arch.name]:
            name = f"{arch.name}_seed{row['seed']}"
            ck,model,device = load_selected(args.output/'confirmation'/f'{name}.pt')
            _,pred = base.evaluate(model,xt,yt,device)
            _,proxy = base.evaluate(model,xt,yt)
            assert torch.equal(pred,proxy)
            matrix = np.zeros((10,10),dtype=np.int64)
            np.add.at(matrix,(yt.numpy(),pred.numpy()),1)
            row.update(test_accuracy=float((pred==yt).sum())/len(yt),proxy_device_agree=True)
            np.savez_compressed(args.output/'confirmation'/f'{name}_test.npz',
                                predictions=pred.numpy(),labels=yt.numpy(),confusion_counts=matrix)
            mask = list(mask_rows(ck['bits'],config,arch))
            assert len({(r['layer_index'],r['x_um'],r['y_um']) for r in mask})==len(mask)
            assert len(mask)==resources(config,arch)['material_sites']
            with (args.output/'confirmation'/f'{name}_write_mask.csv').open('w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f,fieldnames=list(mask[0]))
                w.writeheader();w.writerows(mask)
            print('FINAL_TEST',name,row['test_accuracy'],flush=True)
        del xt
    SAVE(args.output/'confirmation_test_results.json',confirm)
    summary = {}
    for arch in candidates:
        vals = [r['test_accuracy'] for r in confirm if r['architecture']==arch.name]
        summary[arch.name] = dict(mean=float(np.mean(vals)),population_sd=float(np.std(vals)),
                                 seed_values=vals,resources=resources(config,arch))
    SAVE(args.output/'test_summary.json',summary)
    assert all(base.prior.common.hash_file(ROOT/s)==h for s,h in plan['source_hashes'].items())
    status.update(status='complete',elapsed_seconds_this_invocation=time.time()-started)
    SAVE(args.output/'status.json',status)
    print('COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()

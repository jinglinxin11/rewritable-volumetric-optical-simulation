"""Ten binary material planes; 10% per-plane intensity contrast; full erase/rewrite.

Ideal independent intensity channels, not a wave-optical or thermal simulation.
The user-defined layer contrast replaces all former sample-level calibrations.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scheme_a_closed_loop as common

ROOT = Path(__file__).resolve().parent
ARMS = ('binary_offline', 'frozen_binary', 'binary_closed_loop')


@dataclass(frozen=True)
class Config:
    layers: int = 10
    layer_intensity_contrast: float = .10
    write_nm: float = 976.
    read_nm: float = 418.
    nominal_spot_um: float = 50.
    center_pitch_um: float = 50.
    lateral_crosstalk: float = 0.
    interlayer_crosstalk: float = 0.
    write_noise: float = 0.
    read_noise: float = 0.

    def __post_init__(self):
        if not isinstance(self.layers,int) or self.layers<1:
            raise ValueError('Positive integer layer count required')
        if not 0<self.layer_intensity_contrast<1:
            raise ValueError('Contrast must be in (0,1)')
        if any(getattr(self,k)!=0 for k in ('lateral_crosstalk','interlayer_crosstalk','write_noise','read_noise')):
            raise ValueError('This baseline implements zero crosstalk/noise only')
        if any(not math.isfinite(getattr(self,k)) or getattr(self,k)<=0 for k in ('write_nm','read_nm','nominal_spot_um','center_pitch_um')):
            raise ValueError('Positive finite geometry and wavelength metadata required')

    @property
    def shape(self):
        return (self.layers,2,3,9)

    @property
    def minimum_transmission(self):
        return (1-self.layer_intensity_contrast)**self.layers

    @property
    def gain(self):
        return 1/(1-self.minimum_transmission)


def binary_ste(theta):
    probability=theta.sigmoid()
    hard=(theta>=0).to(theta.dtype)
    # Exactly binary forward; derivative is the sigmoid surrogate, not the
    # derivative of a discontinuous physical switching operation.
    return hard+(probability-probability.detach())


def transmissions(bits,config):
    if tuple(bits.shape)!=config.shape:
        raise ValueError('Expected [layer, branch, kernel, coefficient] controls')
    return (1-config.layer_intensity_contrast*bits).prod(0)


def weights(bits,config):
    t=transmissions(bits,config)
    return config.gain*(t[0]-t[1])


class Model(nn.Module):
    def __init__(self,config,seed):
        super().__init__()
        self.config=config
        g=torch.Generator().manual_seed(seed+1000)
        self.theta=nn.Parameter(.5*torch.randn(config.shape,generator=g))
        self.bias=nn.Parameter(torch.zeros(3))
        with torch.random.fork_rng():
            torch.manual_seed(seed+2000)
            self.head=nn.Linear(27,10)

    def bits(self):
        return (self.theta.detach()>=0).float()

    def effective_weights(self):
        return weights(binary_ste(self.theta),self.config)

    def logits_from_weights(self,x,w):
        return common.electronic_logits(self,x@w.T)

    def forward(self,x):
        return self.logits_from_weights(x,self.effective_weights())


class Device(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.register_buffer('state',torch.zeros(config.shape))
        self.initialized=False
        self.program_requests=0
        self.erase_events=0
        self.write_one_events=0
        self.skipped_identical=0
        self.last_erased_ones=0
        self.last_written_ones=0

    @torch.no_grad()
    def program(self,requested):
        if tuple(requested.shape)!=self.config.shape or not bool(((requested==0)|(requested==1)).all()):
            raise ValueError('Device accepts only exact binary masks')
        self.program_requests+=1
        if self.initialized and torch.equal(requested,self.state):
            self.skipped_identical+=1
            return False
        # Clone before clearing, including when caller supplies a state alias.
        target=requested.detach().clone()
        self.last_erased_ones=int(self.state.sum())
        self.state.zero_()
        self.erase_events+=1
        self.last_written_ones=int(target.sum())
        self.write_one_events+=self.last_written_ones
        self.state.copy_(target)
        self.initialized=True
        return True

    @torch.no_grad()
    def effective_weights(self):
        return weights(self.state,self.config)


def training_logits(model,device,x,arm):
    if arm=='binary_offline':
        return model(x)
    observed=x@device.effective_weights().T
    if arm=='binary_closed_loop':
        proxy=x@model.effective_weights().T
        observed=common.surrogate_bridge(observed,proxy)
    elif arm!='frozen_binary':
        raise ValueError(arm)
    return common.electronic_logits(model,observed)


@torch.no_grad()
def evaluate(model,x,y,device=None):
    model.eval()
    w=model.effective_weights() if device is None else device.effective_weights()
    prediction=torch.cat([model.logits_from_weights(part,w).argmax(1) for part in x.split(1024)])
    return float((prediction==y).float().mean()),prediction


def train(arm,seed,args,config,tx,ty,vx,vy):
    model,device=Model(config,seed),Device(config)
    if arm=='frozen_binary': model.theta.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    generator=torch.Generator().manual_seed(seed+3000)
    initial=model.bits().clone()
    if arm!='binary_offline': device.program(initial)
    history,events,snapshots=[],[],[]
    best,checkpoint=-1,None
    for epoch in range(args.epochs+1):
        total,gradient,count=0.,0.,0
        if epoch:
            model.train()
            for ids in torch.randperm(len(ty),generator=generator).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(training_logits(model,device,tx[ids],arm),ty[ids])
                loss.backward()
                gradient+=0 if model.theta.grad is None else float(model.theta.grad.norm())
                optimizer.step()
                total+=float(loss.detach())*len(ids)
                count+=1
                if arm=='binary_closed_loop':
                    did_program=device.program(model.bits())
                    events.append(dict(epoch=epoch,batch=count,rewritten=did_program,erase_events=device.erase_events,
                                       ones_written=device.last_written_ones if did_program else 0,
                                       total_ones_written=device.write_one_events))
        accuracy,_=evaluate(model,vx,vy,None if arm=='binary_offline' else device)
        if accuracy>best:
            best=accuracy
            checkpoint=dict(arm=arm,seed=seed,config=asdict(config),selected_epoch=epoch,
                            validation_accuracy=accuracy,model_state=copy.deepcopy(model.state_dict()),
                            bits=model.bits().clone())
        history.append(dict(epoch=epoch,validation_accuracy=accuracy,training_loss=total/len(ty) if epoch else None,
                            gradient_norm=gradient/max(1,count),changed_bits=int((model.bits()!=initial).sum()),
                            erase_events=device.erase_events,total_ones_written=device.write_one_events))
        if epoch==0 or epoch%5==0 or epoch==args.epochs:
            snapshots.append((epoch,model.bits().numpy().astype(np.uint8),weights(model.bits(),config).numpy()))
            print(f'{arm} seed={seed} epoch={epoch} val={accuracy:.4f} changed={history[-1]["changed_bits"]} erases={device.erase_events}',flush=True)
    name=f'{arm}_seed{seed}'
    torch.save(checkpoint,args.output/(name+'.pt'))
    common.save_json(args.output/(name+'_history.json'),history)
    common.save_json(args.output/(name+'_events.json'),events)
    np.savez_compressed(args.output/(name+'_trajectory.npz'),epochs=[s[0] for s in snapshots],
                        bits=np.stack([s[1] for s in snapshots]),weights=np.stack([s[2] for s in snapshots]))
    return dict(arm=arm,seed=seed,selected_epoch=checkpoint['selected_epoch'],validation_accuracy=best,
                erase_events=device.erase_events,write_one_events=device.write_one_events,
                program_requests=device.program_requests,skipped_identical=device.skipped_identical,
                final_changed_bits=history[-1]['changed_bits'],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_selected(path):
    ck=torch.load(path,weights_only=True)
    config=Config(**ck['config'])
    model=Model(config,ck['seed'])
    model.load_state_dict(ck['model_state'])
    assert torch.equal(model.bits(),ck['bits'])
    device=Device(config)
    device.program(ck['bits'])
    return ck,model,device


def export_mask(path,bits,config):
    import csv
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['layer_index','branch','kernel','coefficient','x_um','y_um','write_bit'])
        for l in range(config.layers):
            for branch in range(2):
                for k in range(3):
                    for j in range(9):
                        x=(3*k+j%3)*config.center_pitch_um
                        y=(3*branch+j//3)*config.center_pitch_um
                        writer.writerow([l,'positive' if branch==0 else 'negative',k,j,x,y,int(bits[l,branch,k,j])])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/scheme_a_binary10_10pct')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--lr',type=float,default=.005)
    args=parser.parse_args()
    if args.output.exists() or min(args.epochs,args.batch_size)<1 or args.lr<=0 or len(set(args.seeds))!=len(args.seeds):
        parser.error('New output path, positive budgets and distinct seeds required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    config=Config();started=time.time()
    sources=['scheme_a_binary10.py','scheme_a_closed_loop.py','scheme_a_round2.py','scheme_a_mnist.py','reproduce.py']
    manifest=dict(status='training',config=asdict(config),arms=list(ARMS),seeds=args.seeds,epochs=args.epochs,
                  batch_size=args.batch_size,learning_rate=args.lr,minimum_transmission=config.minimum_transmission,
                  differential_gain=config.gain,physical_binary_sites=int(np.prod(config.shape)),branch_levels=config.layers+1,
                  source_hashes={p:common.hash_file(ROOT/p) for p in sources},
                  versions=dict(torch=str(torch.__version__),numpy=np.__version__),
                  assumptions=[
                      'User design: ten writable binary layers, 10% relative intensity contrast per layer; not measured ten-layer calibration.',
                      'Old 532nm fit and 20%, 26%, 28.6% sample contrasts are not used.',
                      'Independent serial intensity channels; relative per-layer values 1 and 0.9; common baseline loss factored out.',
                      'Zero lateral/interlayer crosstalk, zero write/read noise; no blur kernel or interpolation.',
                      f'{config.nominal_spot_um:g}um is nominal writing spot size only, not Gaussian FWHM; {config.center_pitch_um:g}um center pitch is a design choice, not validated crosstalk-free packing.',
                      'Wavelengths/pitch/spot are metadata and mask layout, not full-wave propagation; layer z positions unspecified.',
                      'Hard binary forward, sigmoid straight-through surrogate backward. No intermediate optical states in forward.',
                      'Full-volume erase then write every target-one site when a binary mask changes; unchanged masks skip physical programming.',
                      'One initial preparation erase counted. Frozen material programmed once; offline has no training erases.',
                      'Write-site count is not pulse count; erase/write times, thermal kinetics, energy and endurance unspecified.',
                      'Three 3x3 kernels; 14x14 input; electronic sum, bias, ReLU, pooling and 27-to-10 head.',
                      'Ten identical independent layers yield only eleven branch transmission levels, not 1024 distinct levels.',
                      'Offline and closed-loop should coincide in this exact noiseless device; no presumed closed-loop accuracy advantage.',
                      'Same fixed 50k/10k/10k MNIST split; earliest maximum validation selects; all selections precede test.',
                      'Three seeds vary initialization only, not physical devices; official test previously used, not new blind holdout.',
                      'Single deterministic fresh deployment per model; repeated noiseless deployments are not statistical replicates.'])
    common.save_json(args.output/'manifest.json',manifest)
    raw,hashes=common.r2.base.fetch_data(ROOT/'data/MNIST/raw')
    split=np.load(ROOT/'outputs/scheme_a_mnist/split_indices.npz')
    tr,va=split['train'],split['validation']
    assert len(tr)==50000 and len(va)==10000 and len(set(tr)&set(va))==0
    np.savez_compressed(args.output/'split_indices.npz',**{k:split[k] for k in split.files})
    manifest['dataset_hashes']=hashes;common.save_json(args.output/'manifest.json',manifest)
    x=common.r2.patches(raw['train-images-idx3-ubyte.gz'],14)
    y=torch.from_numpy(raw['train-labels-idx1-ubyte.gz'].astype(np.int64))
    tx,ty,vx,vy=x[tr],y[tr],x[va],y[va];del x
    rows=[]
    for seed in args.seeds:
        for arm in ARMS:
            rows.append(train(arm,seed,args,config,tx,ty,vx,vy))
            common.save_json(args.output/'validation_results.json',rows)
    common.save_json(args.output/'selection_before_test.json',rows)
    del tx,ty,vx,vy
    manifest['status']='testing';common.save_json(args.output/'manifest.json',manifest)
    xt=common.r2.patches(raw['t10k-images-idx3-ubyte.gz'],14)
    yt=torch.from_numpy(raw['t10k-labels-idx1-ubyte.gz'].astype(np.int64))
    for row in rows:
        name=f"{row['arm']}_seed{row['seed']}"
        ck,model,device=load_selected(args.output/(name+'.pt'))
        acc,pred=evaluate(model,xt,yt,device)
        row.update(test_accuracy=acc,deployment_erase_events=device.erase_events,deployment_write_ones=device.write_one_events)
        np.savez_compressed(args.output/(name+'_test.npz'),predictions=pred.numpy(),labels=yt.numpy())
        export_mask(args.output/(name+'_write_mask.csv'),ck['bits'],config)
        print('TEST',name,acc,flush=True)
    common.save_json(args.output/'results.json',rows)
    summary={}
    for arm in ARMS:
        values=[r['test_accuracy'] for r in rows if r['arm']==arm]
        summary[arm]=dict(mean=float(np.mean(values)),population_sd=float(np.std(values)),seed_values=values)
    common.save_json(args.output/'summary.json',summary)
    assert all(common.hash_file(ROOT/p)==h for p,h in manifest['source_hashes'].items())
    manifest.update(status='complete',elapsed_seconds=time.time()-started)
    common.save_json(args.output/'manifest.json',manifest)
    print('COMPLETE',manifest['elapsed_seconds'],flush=True)


if __name__=='__main__':
    main()

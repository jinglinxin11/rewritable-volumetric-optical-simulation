"""Binary material convolution AND classifier, with atomic full-volume rewriting.

Ideal intensity-channel simulation. Re-encoding, differential summation, bias,
activation, pooling and optimization are electronic, not all-optical operations.
"""
import argparse
import copy
import csv
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import scheme_a_binary10 as prior

ROOT=Path(__file__).resolve().parent
ARMS=('joint_offline','frozen_material_head','joint_closed_loop')


def shapes(config):
    return {'conv':(config.layers,2,3,9),'head':(config.layers,2,10,27)}


def optical_weights(bits,config):
    t=(1-config.layer_intensity_contrast*bits).prod(0)
    return config.gain*(t[0]-t[1])


def feature_vector(x,w,bias):
    z=x@w.T
    h=F.relu(z.transpose(1,2).reshape(-1,3,12,12)+bias[None,:,None,None])
    return F.avg_pool2d(h,4).flatten(1)


def encode_features(h):
    # Normalize each sample to realizable relative light intensities [0, 1].
    # Electronic rescaling after detection preserves the intended linear head.
    scale=h.amax(1,keepdim=True).clamp_min(1.)
    return h/scale,scale


def material_classifier(h,w,bias):
    illumination,scale=encode_features(h)
    return (illumination@w.T)*scale+bias


class Model(nn.Module):
    def __init__(self,config,seed):
        super().__init__()
        self.config=config
        for zone,offset in (('conv',1000),('head',4000)):
            g=torch.Generator().manual_seed(seed+offset)
            setattr(self,zone+'_theta',nn.Parameter(.5*torch.randn(shapes(config)[zone],generator=g)))
        self.conv_bias=nn.Parameter(torch.zeros(3))
        # Match the previous electronic head bias initialization, but retain no
        # electronic classifier weight matrix anywhere in this model.
        previous=prior.Model(config,seed)
        self.head_bias=nn.Parameter(previous.head.bias.detach().clone())

    def bits(self):
        return {zone:(getattr(self,zone+'_theta').detach()>=0).float() for zone in ('conv','head')}

    def weights(self):
        return {zone:optical_weights(prior.binary_ste(getattr(self,zone+'_theta')),self.config) for zone in ('conv','head')}

    def logits_from_weights(self,x,w):
        h=feature_vector(x,w['conv'],self.conv_bias)
        return material_classifier(h,w['head'],self.head_bias)

    def forward(self,x):
        return self.logits_from_weights(x,self.weights())


class Device(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        for zone,shape in shapes(config).items(): self.register_buffer(zone+'_state',torch.zeros(shape))
        self.initialized=False
        self.requests=0;self.erase_events=0;self.skipped_identical=0
        self.write_ones={'conv':0,'head':0}
        self.last_written={'conv':0,'head':0}

    @torch.no_grad()
    def program(self,target):
        if set(target)!={'conv','head'}: raise ValueError('Both zones required for whole-volume programming')
        for zone,shape in shapes(self.config).items():
            v=target[zone]
            if tuple(v.shape)!=shape or not bool(((v==0)|(v==1)).all()):
                raise ValueError('Exact binary masks with correct shapes required')
        self.requests+=1
        if self.initialized and all(torch.equal(target[z],getattr(self,z+'_state')) for z in target):
            self.skipped_identical+=1;self.last_written={'conv':0,'head':0};return False
        copied={z:v.detach().clone() for z,v in target.items()}
        # Erase BOTH regions before writing either. No delta-only updates.
        for z in copied: getattr(self,z+'_state').zero_()
        self.erase_events+=1
        for z,v in copied.items():
            self.last_written[z]=int(v.sum());self.write_ones[z]+=self.last_written[z]
            getattr(self,z+'_state').copy_(v)
        self.initialized=True
        return True

    @torch.no_grad()
    def weights(self):
        return {z:optical_weights(getattr(self,z+'_state'),self.config) for z in ('conv','head')}


def training_logits(model,device,x,arm):
    proxy=model.weights()
    if arm=='joint_offline': return model.logits_from_weights(x,proxy)
    if arm not in ARMS: raise ValueError(arm)
    physical=device.weights()
    # Preserve propagation of gradients through the head to convolution features.
    # Device states are detached; the proxy supplies weight Jacobians only.
    bridged={z:prior.common.surrogate_bridge(physical[z],proxy[z]) for z in proxy}
    return model.logits_from_weights(x,bridged)


@torch.no_grad()
def evaluate(model,x,y,device=None):
    model.eval();w=model.weights() if device is None else device.weights()
    prediction=torch.cat([model.logits_from_weights(p,w).argmax(1) for p in x.split(1024)])
    return float((prediction==y).float().mean()),prediction


def train(arm,seed,args,config,tx,ty,vx,vy):
    model,device=Model(config,seed),Device(config)
    if arm=='frozen_material_head': model.head_theta.requires_grad_(False)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    generator=torch.Generator().manual_seed(seed+3000)
    initial={z:v.clone() for z,v in model.bits().items()}
    if arm!='joint_offline': device.program(initial)
    history,events,snapshots=[],[],[];best=-1.;checkpoint=None
    for epoch in range(args.epochs+1):
        loss_sum=0.;grads={'conv':0.,'head':0.};count=0
        if epoch:
            model.train()
            for ids in torch.randperm(len(ty),generator=generator).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(training_logits(model,device,tx[ids],arm),ty[ids])
                loss.backward()
                for z in grads:
                    g=getattr(model,z+'_theta').grad
                    grads[z]+=0. if g is None else float(g.norm())
                optimizer.step();count+=1;loss_sum+=float(loss.detach())*len(ids)
                if arm!='joint_offline':
                    changed=device.program(model.bits())
                    events.append(dict(epoch=epoch,batch=count,rewritten=changed,erase_events=device.erase_events,
                                       conv_ones=device.last_written['conv'],head_ones=device.last_written['head']))
        acc,_=evaluate(model,vx,vy,None if arm=='joint_offline' else device)
        bits=model.bits()
        if acc>best:
            best=acc
            checkpoint=dict(arm=arm,seed=seed,config=asdict(config),selected_epoch=epoch,validation_accuracy=acc,
                            model_state=copy.deepcopy(model.state_dict()),bits={z:v.clone() for z,v in bits.items()})
        record=dict(epoch=epoch,validation_accuracy=acc,training_loss=loss_sum/len(ty) if epoch else None,
                    erase_events=device.erase_events,write_ones=dict(device.write_ones))
        for z in grads:
            record[z+'_gradient_norm']=grads[z]/max(count,1)
            record[z+'_changed_bits']=int((bits[z]!=initial[z]).sum())
        history.append(record)
        if epoch==0 or epoch%5==0 or epoch==args.epochs:
            snapshots.append(dict(epoch=epoch,**{z:bits[z].numpy().astype(np.uint8) for z in bits}))
            print(f'{arm} seed={seed} epoch={epoch} val={acc:.4f} conv_bits={record["conv_changed_bits"]} head_bits={record["head_changed_bits"]} erases={device.erase_events}',flush=True)
    name=f'{arm}_seed{seed}'
    torch.save(checkpoint,args.output/(name+'.pt'))
    prior.common.save_json(args.output/(name+'_history.json'),history)
    prior.common.save_json(args.output/(name+'_events.json'),events)
    np.savez_compressed(args.output/(name+'_trajectory.npz'),epochs=[s['epoch'] for s in snapshots],
                        conv=np.stack([s['conv'] for s in snapshots]),head=np.stack([s['head'] for s in snapshots]))
    return dict(arm=arm,seed=seed,selected_epoch=checkpoint['selected_epoch'],validation_accuracy=best,
                erase_events=device.erase_events,write_ones=dict(device.write_ones),program_requests=device.requests,
                skipped_identical=device.skipped_identical,conv_changed_bits=history[-1]['conv_changed_bits'],
                head_changed_bits=history[-1]['head_changed_bits'],
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))


def load_selected(path):
    ck=torch.load(path,weights_only=True);config=prior.Config(**ck['config'])
    model=Model(config,ck['seed']);model.load_state_dict(ck['model_state'])
    for z,v in model.bits().items(): assert torch.equal(v,ck['bits'][z])
    device=Device(config);device.program(ck['bits'])
    return ck,model,device


def mask_rows(bits,config):
    for layer in range(config.layers):
        for z in ('conv','head'):
            v=bits[z][layer]
            for branch in range(2):
                for out in range(v.shape[1]):
                    for k in range(v.shape[2]):
                        if z=='conv': col=9*out+k;row=branch
                        else: col=k;row=2+branch*10+out
                        yield dict(layer_index=layer,zone=z,branch=branch,output_index=out,input_index=k,
                                   x_um=col*config.center_pitch_um,y_um=row*config.center_pitch_um,
                                   write_bit=int(v[branch,out,k]))


def export_mask(path,bits,config):
    rows=list(mask_rows(bits,config))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/scheme_a_material_head_binary10')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--lr',type=float,default=.005)
    args=p.parse_args()
    if args.output.exists() or min(args.epochs,args.batch_size)<1 or args.lr<=0 or len(set(args.seeds))!=len(args.seeds):
        p.error('New output directory, positive budgets and unique seeds required')
    args.output.mkdir(parents=True);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    config=prior.Config();started=time.time()
    sources=['scheme_a_material_head.py','scheme_a_binary10.py','scheme_a_closed_loop.py','scheme_a_round2.py','scheme_a_mnist.py','reproduce.py']
    manifest=dict(status='training',config=asdict(config),arms=list(ARMS),seeds=args.seeds,epochs=args.epochs,
                  batch_size=args.batch_size,learning_rate=args.lr,physical_sites={'conv':540,'head':5400,'total':5940},
                  electronic_trainable_parameters=13,source_hashes={s:prior.common.hash_file(ROOT/s) for s in sources},
                  versions=dict(torch=str(torch.__version__),numpy=np.__version__),
                  protocol='50k/10k/10k fixed split; 14x14 input; 3 kernels; 27-to-10 material head; from scratch; earliest maximum validation; all selections before test.',
                  assumptions=[
                      'Ten binary layers per channel; relative intensity 1 or 0.9; old sample contrasts and 532nm calibration not used.',
                      'Zero noise/crosstalk; independent intensity products, not coherent propagation or physical ten-layer measurements.',
                      'Hard binary forward in both banks; sigmoid surrogate gradients; biases remain electronic.',
                      '27 nonnegative pooled features re-encoded as h/max(1,max(h)); electronic output rescaling restores the same scores.',
                      'Re-encoding is an ideal amplitude/intensity interface; no finite DAC/ADC, illumination power, bandwidth, latency or saturation model.',
                      'No digital classifier weight matrix; only 3 convolution biases and 10 class biases remain electronically trainable.',
                      'Frozen-head control fixes a random material classifier but learns convolution and biases; it is not a parameter-count-matched control.',
                      'Offline and closed-loop should agree exactly under the zero-mismatch model; each uses binary weights in training and inference.',
                      'Both material banks share one whole-volume erase; rewrite all target ones even in an unchanged bank.',
                      'Identical complete masks skip physical programming; initial preparation and final deployment each count an erase.',
                      'Write-site counts are not laser pulse counts; no erase temperature, time, energy, kinetics or endurance model.',
                      f'{config.nominal_spot_um:g}um nominal spot and {config.center_pitch_um:g}um center pitch; one 27-column x 22-row layout per physical layer; no blur kernel or validated packing-crosstalk model.',
                      '976nm/418nm metadata; no layer z coordinates or refractive propagation model; mask CSV not an exposure recipe.',
                      '54 convolution intensity products per window x 144 windows, plus 540 head products per image: 8316 channel readings if products read separately.',
                      'Ten layers encode 11 levels per branch, not 1024. Two readout stages are not twenty stacked layers.',
                      'Three initialization seeds only, not device replicates. Previously used official test set, not a new blind holdout.',
                      'Previous electronic-head baseline is historical same-budget reference; head initialization, precision and parameterization differ.'])
    prior.common.save_json(args.output/'manifest.json',manifest)
    raw,hashes=prior.common.r2.base.fetch_data(ROOT/'data/MNIST/raw')
    splits=np.load(ROOT/'outputs/scheme_a_mnist/split_indices.npz');tr,va=splits['train'],splits['validation']
    assert len(tr)==50000 and len(va)==10000 and len(set(tr)&set(va))==0
    np.savez_compressed(args.output/'split_indices.npz',**{k:splits[k] for k in splits.files})
    manifest['dataset_hashes']=hashes;prior.common.save_json(args.output/'manifest.json',manifest)
    x=prior.common.r2.patches(raw['train-images-idx3-ubyte.gz'],14)
    y=torch.from_numpy(raw['train-labels-idx1-ubyte.gz'].astype(np.int64))
    tx,ty,vx,vy=x[tr],y[tr],x[va],y[va];del x
    rows=[]
    for seed in args.seeds:
        for arm in ARMS:
            rows.append(train(arm,seed,args,config,tx,ty,vx,vy))
            prior.common.save_json(args.output/'validation_results.json',rows)
    prior.common.save_json(args.output/'selection_before_test.json',rows)
    del tx,ty,vx,vy
    manifest['status']='testing';prior.common.save_json(args.output/'manifest.json',manifest)
    xt=prior.common.r2.patches(raw['t10k-images-idx3-ubyte.gz'],14)
    yt=torch.from_numpy(raw['t10k-labels-idx1-ubyte.gz'].astype(np.int64))
    for row in rows:
        name=f"{row['arm']}_seed{row['seed']}";ck,model,device=load_selected(args.output/(name+'.pt'))
        acc,pred=evaluate(model,xt,yt,device)
        row.update(test_accuracy=acc,deployment_erases=device.erase_events,deployment_write_ones=dict(device.write_ones))
        np.savez_compressed(args.output/(name+'_test.npz'),predictions=pred.numpy(),labels=yt.numpy())
        export_mask(args.output/(name+'_write_mask.csv'),ck['bits'],config)
        print('TEST',name,acc,flush=True)
    prior.common.save_json(args.output/'results.json',rows)
    summary={}
    for arm in ARMS:
        vals=[r['test_accuracy'] for r in rows if r['arm']==arm]
        summary[arm]=dict(mean=float(np.mean(vals)),population_sd=float(np.std(vals)),seed_values=vals)
    prior.common.save_json(args.output/'summary.json',summary)
    assert all(prior.common.hash_file(ROOT/s)==h for s,h in manifest['source_hashes'].items())
    manifest.update(status='complete',elapsed_seconds=time.time()-started);prior.common.save_json(args.output/'manifest.json',manifest)
    print('COMPLETE',manifest['elapsed_seconds'],flush=True)


if __name__=='__main__': main()

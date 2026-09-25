import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import scheme_a_material_head as m


class MaterialHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_parameter_budget_no_electronic_matrix(self):
        model=m.Model(m.prior.Config(),0)
        self.assertEqual(sum(p.numel() for p in model.parameters()),5953)
        self.assertEqual(set(dict(model.named_parameters())),{'conv_theta','head_theta','conv_bias','head_bias'})
        self.assertFalse(any(isinstance(mod,torch.nn.Linear) for mod in model.modules()))

    def test_encoding_bounded_and_equivalent(self):
        h=torch.tensor([[0.]*27,[10.]*27,[.2]*27],requires_grad=True)
        w=torch.randn(10,27,requires_grad=True);bias=torch.randn(10)
        u,s=m.encode_features(h)
        self.assertTrue(bool((u>=0).all() and (u<=1).all()))
        torch.testing.assert_close(u*s,h)
        torch.testing.assert_close(m.material_classifier(h,w,bias),h@w.T+bias)
        m.material_classifier(h,w,bias).sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_explicit_head_channel_products(self):
        c=m.prior.Config();model=m.Model(c,0);bits=model.bits()['head']
        t=(1-.1*bits).prod(0);h=torch.rand(5,27)*12
        u,s=m.encode_features(h)
        signals=u[:,None,None,:]*t[None,:,:,:]
        explicit=c.gain*(signals[:,0]-signals[:,1]).sum(-1)*s+model.head_bias
        torch.testing.assert_close(explicit,m.material_classifier(h,m.optical_weights(bits,c),model.head_bias))

    def test_atomic_erase_rewrites_both_regions(self):
        c=m.prior.Config();d=m.Device(c);target={z:torch.zeros(shape) for z,shape in m.shapes(c).items()}
        target['conv'][0,0,0,0]=1;target['head'][0,0,0,0]=1;d.program(target)
        target['head'][0,0,0,1]=1;d.program(target)
        self.assertEqual(d.erase_events,2)
        self.assertEqual(d.write_ones,{'conv':2,'head':3})
        self.assertEqual(d.last_written,{'conv':1,'head':2})
        self.assertFalse(d.program(target));self.assertEqual(d.last_written,{'conv':0,'head':0})
        before=d.head_state.clone()
        with self.assertRaises(ValueError): d.program({'conv':target['conv'],'head':target['head']+.1})
        self.assertTrue(torch.equal(before,d.head_state))

    def test_both_bank_gradients_and_exact_offline_closed(self):
        c=m.prior.Config();a=m.Model(c,1);b=m.Model(c,1);d=m.Device(c);d.program(b.bits())
        x=torch.rand(4,144,9);labels=torch.arange(4)
        la=m.training_logits(a,d,x,'joint_offline');lb=m.training_logits(b,d,x,'joint_closed_loop')
        torch.testing.assert_close(la,lb,rtol=0,atol=0)
        torch.nn.functional.cross_entropy(la,labels).backward();torch.nn.functional.cross_entropy(lb,labels).backward()
        for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
        self.assertGreater(float(b.conv_theta.grad.norm()),0);self.assertGreater(float(b.head_theta.grad.norm()),0)
        self.assertFalse(d.head_state.requires_grad)

    def test_5940_mask_positions_no_collisions(self):
        c=m.prior.Config();rows=list(m.mask_rows(m.Model(c,0).bits(),c))
        self.assertEqual(len(rows),5940)
        self.assertEqual(len({(r['layer_index'],r['x_um'],r['y_um']) for r in rows}),5940)
        self.assertEqual(sum(r['zone']=='head' for r in rows),5400)

    def test_default_pitch_equals_nominal_diameter(self):
        c=m.prior.Config()
        self.assertEqual(c.nominal_spot_um,50.)
        self.assertEqual(c.center_pitch_um,c.nominal_spot_um)
        bits=m.Model(c,0).bits()
        rows=list(m.mask_rows(bits,c))
        xs=sorted({r['x_um'] for r in rows})
        ys=sorted({r['y_um'] for r in rows})
        self.assertEqual(xs,[50.*i for i in range(27)])
        self.assertEqual(ys,[50.*i for i in range(22)])
        old_rows=list(m.mask_rows(bits,m.prior.Config(center_pitch_um=100.)))
        for new,old in zip(rows,old_rows):
            self.assertEqual(new['x_um'],old['x_um']/2)
            self.assertEqual(new['y_um'],old['y_um']/2)
            self.assertEqual(new['write_bit'],old['write_bit'])

    def test_pitch_metadata_does_not_change_numerical_forward(self):
        new=m.Model(m.prior.Config(),0)
        old=m.Model(m.prior.Config(center_pitch_um=100.),0)
        g=torch.Generator().manual_seed(17)
        x=torch.rand(3,144,9,generator=g)
        torch.testing.assert_close(new(x),old(x),rtol=0,atol=0)
        new(x).square().sum().backward()
        old(x).square().sum().backward()
        for a,b in zip(new.parameters(),old.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)

    def test_persistent_banks_do_not_follow_optimizer(self):
        c=m.prior.Config();model=m.Model(c,0);d=m.Device(c);d.program(model.bits())
        before={z:getattr(d,z+'_state').clone() for z in ('conv','head')}
        with torch.no_grad():model.conv_theta.mul_(-1);model.head_theta.mul_(-1)
        for z in before:self.assertTrue(torch.equal(before[z],getattr(d,z+'_state')))

    def test_small_train_reload_and_frozen_head(self):
        g=torch.Generator().manual_seed(59);x=torch.rand(16,144,9,generator=g);y=torch.arange(16)%10
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(output=Path(folder),epochs=2,batch_size=8,lr=.005)
            for arm in m.ARMS:
                row=m.train(arm,0,args,m.prior.Config(),x,y,x,y)
                ck,model,d=m.load_selected(args.output/(arm+'_seed0.pt'))
                a,p=m.evaluate(model,x,y,d);aa,pp=m.evaluate(model,x,y)
                self.assertEqual(a,aa);self.assertTrue(torch.equal(p,pp))
                for bits in ck['bits'].values():self.assertTrue(bool(((bits==0)|(bits==1)).all()))
                if arm=='frozen_material_head':self.assertEqual(row['head_changed_bits'],0)


if __name__=='__main__':unittest.main()

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import torch.nn.functional as F
import scheme_a_conv_search as m


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_baseline_exact_forward_and_gradient(self):
        c,a = m.base.prior.Config(),m.Architecture()
        new,old = m.Model(c,a,1),m.previous.Model(c,1)
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,new.state_dict()[key]))
        x = torch.rand(16,144,9)
        p,q = new(x),old(x)
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        p.sum().backward();q.sum().backward()
        for pa,pb in zip(new.parameters(),old.parameters()):
            torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)

    def test_unfold_matches_explicit_patch_and_conv(self):
        raw = np.arange(2*28*28,dtype=np.int64).reshape(2,28,28).astype(np.uint8)
        img = F.interpolate(torch.from_numpy(raw).float()[:,None]/255,size=(14,14),mode='area')
        for a in m.GRID:
            x = m.patches(raw,a)
            self.assertEqual(tuple(x.shape),(2,a.side**2,a.kernel**2))
            torch.testing.assert_close(x[0,0],img[0,0,:a.kernel,:a.kernel].flatten())
            model = m.Model(m.base.prior.Config(),a,0)
            w = model.weights()['conv']
            z = (x@w.T).transpose(1,2).reshape(2,a.channels,a.side,a.side)
            expected = F.conv2d(img,w.reshape(a.channels,1,a.kernel,a.kernel),stride=a.stride)
            torch.testing.assert_close(z,expected,atol=2e-6,rtol=2e-5)

    def test_all_shapes_gradients_and_budget(self):
        c = m.base.prior.Config()
        for a in m.GRID:
            model,device = m.Model(c,a,0),m.Device(c,a)
            device.program(model.bits())
            x = torch.rand(16,a.side**2,a.kernel**2)
            p = m.previous.training_logits(model,device,x)
            self.assertEqual(tuple(p.shape),(16,10))
            torch.testing.assert_close(p,model(x),rtol=0,atol=0)
            F.cross_entropy(p,torch.arange(16)%10).backward()
            for z in m.ZONES:
                g = getattr(model,z+'_theta').grad
                self.assertTrue(bool(torch.isfinite(g).all()))
                self.assertGreater(float(g.norm()),0)
            self.assertEqual(sum(p.numel() for p in model.parameters()),m.resources(c,a)['trainable_parameters'])
            self.assertFalse(any(isinstance(x,torch.nn.Linear) for x in model.modules()))

    def test_atomic_all_banks_and_coordinate_uniqueness(self):
        c,a = m.base.prior.Config(),m.Architecture(12,5,2)
        model,d = m.Model(c,a,0),m.Device(c,a)
        bits = model.bits()
        d.program(bits)
        bits['conv'][0,0,0,0] = 1-bits['conv'][0,0,0,0]
        d.program(bits)
        for z in m.ZONES:
            self.assertEqual(d.last_written[z],int(bits[z].sum()))
        self.assertFalse(d.program(bits))
        with self.assertRaises(ValueError):
            d.program(dict(bits,fc1=bits['fc1']+.1))
        rows = list(m.mask_rows(bits,c,a))
        self.assertEqual(len(rows),m.resources(c,a)['material_sites'])
        self.assertEqual(len(rows),len({(r['layer_index'],r['x_um'],r['y_um']) for r in rows}))

    def test_train_select_reload(self):
        c,a = m.base.prior.Config(),m.Architecture(6,5,2)
        x,y = torch.rand(24,a.side**2,a.kernel**2),torch.arange(24)%10
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(epochs=2,batch_size=8,lr=.005)
            row = m.train(c,a,0,args,Path(directory),x,y,x,y)
            ck,model,d = m.load_selected(Path(directory)/f'{a.name}_seed0.pt')
            self.assertEqual(row['selected_epoch'],ck['selected_epoch'])
            torch.testing.assert_close(model(x),model.logits_from_weights(x,d.weights()),rtol=0,atol=0)
            self.assertEqual(m.train(c,a,0,args,Path(directory),x,y,x,y),row)


if __name__=='__main__':
    unittest.main()

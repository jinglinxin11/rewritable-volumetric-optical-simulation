import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
import scheme_a_binary10 as b


class BinaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_endpoints_and_count_levels(self):
        c=b.Config()
        for n in range(11):
            bits=torch.zeros(c.shape);bits[:n]=1
            t=b.transmissions(bits,c)
            torch.testing.assert_close(t,torch.full_like(t,.9**n))
        self.assertAlmostEqual(c.minimum_transmission,.3486784401)
        self.assertEqual(int(np.prod(c.shape)),540)

    def test_layer_order_does_not_change_weights(self):
        c=b.Config();bits=b.Model(c,0).bits()
        torch.testing.assert_close(b.weights(bits,c),b.weights(bits.flip(0),c))

    def test_binary_forward_surrogate_gradient(self):
        theta=torch.tensor([-1.,0.,1.],requires_grad=True)
        out=b.binary_ste(theta)
        self.assertEqual(out.tolist(),[0.,1.,1.])
        out.sum().backward()
        torch.testing.assert_close(theta.grad,theta.sigmoid()*(1-theta.sigmoid()))

    def test_full_erase_rewrites_unchanged_one_sites(self):
        c=b.Config();device=b.Device(c);bits=torch.zeros(c.shape)
        bits[0,0,0,0]=1;device.program(bits)
        bits[0,0,0,1]=1;device.program(bits)
        self.assertEqual(device.erase_events,2)
        self.assertEqual(device.last_erased_ones,1)
        self.assertEqual(device.last_written_ones,2)
        self.assertEqual(device.write_one_events,3)
        self.assertFalse(device.program(bits))
        self.assertEqual(device.erase_events,2)
        self.assertFalse(device.state.requires_grad)
        with self.assertRaises(ValueError): device.program(torch.full(c.shape,.5))

    def test_offline_closed_loop_exact_forward_and_gradients(self):
        c=b.Config();a=b.Model(c,4);d=b.Model(c,4);dev=b.Device(c);dev.program(d.bits())
        x=torch.rand(4,144,9)
        la=b.training_logits(a,dev,x,'binary_offline')
        ld=b.training_logits(d,dev,x,'binary_closed_loop')
        torch.testing.assert_close(la,ld,rtol=0,atol=0)
        la.square().sum().backward();ld.square().sum().backward()
        for pa,pd in zip(a.parameters(),d.parameters()): torch.testing.assert_close(pa.grad,pd.grad,rtol=0,atol=0)

    def test_differential_range_and_explicit_products(self):
        c=b.Config();bits=torch.ones(c.shape);bits[:,0]=0
        torch.testing.assert_close(b.weights(bits,c),torch.ones(3,9))
        x=torch.rand(2,144,9);t=b.transmissions(bits,c)
        explicit=(x[:,:,None,:]*(t[0]-t[1])[None,None,:,:]).sum(-1)*c.gain
        torch.testing.assert_close(explicit,x@b.weights(bits,c).T)

    def test_persistent_state(self):
        c=b.Config();model=b.Model(c,1);dev=b.Device(c);dev.program(model.bits())
        before=dev.state.clone()
        with torch.no_grad(): model.theta.mul_(-1)
        self.assertTrue(torch.equal(before,dev.state))
        dev.program(model.bits());self.assertTrue(torch.equal(model.bits(),dev.state))

    def test_small_training_and_reload(self):
        c=b.Config();x=torch.rand(12,144,9);y=torch.arange(12)%10
        with tempfile.TemporaryDirectory() as folder:
            args=SimpleNamespace(output=Path(folder),epochs=2,batch_size=6,lr=.005)
            for arm in b.ARMS:
                row=b.train(arm,0,args,c,x,y,x,y)
                ck,model,dev=b.load_selected(args.output/(arm+'_seed0.pt'))
                a,p=b.evaluate(model,x,y,dev);aa,pp=b.evaluate(model,x,y)
                self.assertEqual(a,aa);self.assertTrue(torch.equal(p,pp))
                self.assertEqual(dev.erase_events,1)
                if arm=='frozen_binary':
                    self.assertEqual(row['final_changed_bits'],0)
                    self.assertEqual(row['erase_events'],1)


if __name__=='__main__': unittest.main()

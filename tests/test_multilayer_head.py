"""Tests for the additional material banks and electronic inter-layer interfaces."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import scheme_a_multilayer_head as m


class MultilayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_budget(self):
        model = m.Model(m.base.prior.Config(), 0)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 31321)
        self.assertFalse(any(isinstance(layer, torch.nn.Linear) for layer in model.modules()))
        self.assertEqual(sum(v.numel() for v in model.bits().values()), 31260)

    def test_explicit_physical_forward_and_encoding(self):
        c = m.base.prior.Config()
        model = m.Model(c, 0)
        x = torch.rand(8, 144, 9)
        bits = model.bits()
        h = m.base.feature_vector(x, m.base.optical_weights(bits['conv'], c), model.conv_bias)
        for i in range(1, 4):
            u, scale = m.base.encode_features(h)
            self.assertTrue(bool(((u >= 0) & (u <= 1)).all()))
            transmission = (1-.1*bits[f'fc{i}']).prod(0)
            signals = u[:, None, None, :] * transmission[None, :, :, :]
            h = c.gain*(signals[:, 0]-signals[:, 1]).sum(-1)*scale+getattr(model, f'fc{i}_bias')
            if i < 3:
                h = h.relu()
        torch.testing.assert_close(h, model(x), atol=2e-6, rtol=2e-5)

    def test_all_banks_receive_gradients_and_exact_proxy(self):
        c = m.base.prior.Config()
        a, b, device = m.Model(c, 1), m.Model(c, 1), m.Device(c)
        device.program(b.bits())
        x = torch.rand(16, 144, 9)
        y = torch.arange(16) % 10
        la, lb = a(x), m.training_logits(b, device, x)
        torch.testing.assert_close(la, lb, atol=0, rtol=0)
        torch.nn.functional.cross_entropy(la, y).backward()
        torch.nn.functional.cross_entropy(lb, y).backward()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=0, rtol=0)
        for z in m.ZONES:
            self.assertGreater(float(getattr(b, z+'_theta').grad.norm()), 0)
            self.assertFalse(getattr(device, z+'_state').requires_grad)

    def test_atomic_rewrite_and_invalid_masks(self):
        c = m.base.prior.Config()
        d, model = m.Device(c), m.Model(c, 0)
        bits = model.bits()
        d.program(bits)
        bits['fc3'][0, 0, 0, 0] = 1-bits['fc3'][0, 0, 0, 0]
        self.assertTrue(d.program(bits))
        self.assertEqual(d.erase_events, 2)
        for z in m.ZONES:
            self.assertEqual(d.last_written[z], int(bits[z].sum()))
        self.assertFalse(d.program(bits))
        before = {z: getattr(d,z+'_state').clone() for z in m.ZONES}
        bad = dict(bits, fc1=bits['fc1']+.1)
        with self.assertRaises(ValueError):
            d.program(bad)
        for z in m.ZONES:
            self.assertTrue(torch.equal(before[z], getattr(d,z+'_state')))

    def test_layout_unique(self):
        c = m.base.prior.Config()
        rows = list(m.mask_rows(m.Model(c, 0).bits(), c))
        self.assertEqual(len(rows), 31260)
        self.assertEqual(len({(r['layer_index'], r['x_um'], r['y_um']) for r in rows}), 31260)

    def test_train_reload(self):
        x, y = torch.rand(32, 144, 9), torch.arange(32) % 10
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), epochs=2, batch_size=8, lr=.005)
            m.train(0, args, m.base.prior.Config(), x, y, x, y)
            ck, model, device = m.load_selected(args.output/'three_layer_seed0.pt')
            a, pa = m.base.evaluate(model, x, y, device)
            b, pb = m.base.evaluate(model, x, y)
            self.assertEqual(a, b)
            self.assertTrue(torch.equal(pa, pb))
            self.assertEqual(ck['widths'], [27, 32, 16, 10])


if __name__ == '__main__':
    unittest.main()

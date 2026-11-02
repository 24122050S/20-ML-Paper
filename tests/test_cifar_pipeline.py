"""Kiểm tra protocol, preprocessing và ResNet bằng dữ liệu nhỏ không cần mạng."""

from pathlib import Path
import copy
import importlib.util
import json
import tempfile
import unittest

import numpy as np

HAS_CIFAR = all(importlib.util.find_spec(name) is not None for name in
                ("torch", "huggingface_hub", "pyarrow", "PIL", "threadpoolctl"))
if HAS_CIFAR:
    import torch
    from torch import nn
    from src.cifar_data import CifarArrays, CifarSubset, make_protocol
    from src.resnet import ResNet32, freeze_backbone
    from src.cifar_smoke import extract_features, make_loader, validate_checkpoint


@unittest.skipUnless(HAS_CIFAR, "Cài requirements-cifar.txt để chạy test pipeline ảnh")
class CifarPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_protocol_is_bijective_and_phases_are_disjoint(self):
        for seed in (7, 42, 2026):
            with self.subTest(seed=seed):
                protocol = make_protocol(seed, 5)
                order = np.asarray(protocol["class_order"])
                mapping = np.asarray(protocol["original_to_global"])
                np.testing.assert_array_equal(mapping[order], np.arange(100))
                groups = protocol["phase_global_classes"]
                self.assertEqual([len(g) for g in groups], [50, 10, 10, 10, 10, 10])
                self.assertEqual(sorted(x for group in groups for x in group), list(range(100)))
        self.assertEqual(make_protocol(42), make_protocol(42))
        self.assertNotEqual(make_protocol(42)["class_order"], make_protocol(7)["class_order"])

    def test_invalid_phase_count_is_rejected(self):
        for count in (0, 3, 6, 51):
            with self.subTest(count=count), self.assertRaises(ValueError):
                make_protocol(n_incremental_phases=count)

    def test_subset_preserves_global_label_and_normalization(self):
        images = np.full((3, 32, 32, 3), 255, dtype=np.uint8)
        labels = np.array([4, 7, 9])
        mapping = np.arange(100)[::-1].copy()
        dataset = CifarSubset(images, labels, [1], mapping, [0.5]*3, [0.25]*3)
        tensor, label = dataset[0]
        self.assertEqual(label, 92)
        self.assertEqual(tensor.shape, (3, 32, 32))
        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2.0))

    def test_eval_is_deterministic_and_augmentation_is_seeded(self):
        image = np.arange(32*32*3, dtype=np.uint16).astype(np.uint8).reshape(1,32,32,3)
        args = (image, np.array([0]), [0], np.arange(100), [0.5]*3, [0.25]*3)
        evaluation = CifarSubset(*args)
        torch.testing.assert_close(evaluation[0][0], evaluation[0][0], rtol=0, atol=0)
        training = CifarSubset(*args, augment=True)
        torch.manual_seed(9)
        first = training[0][0]
        torch.manual_seed(9)
        torch.testing.assert_close(first, training[0][0], rtol=0, atol=0)

    def test_resnet_depth_features_and_training_update(self):
        torch.manual_seed(42)
        model = ResNet32(n_classes=50)
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 31)
        self.assertEqual(sum(isinstance(m, nn.Linear) for m in model.modules()), 1)
        images = torch.randn(4,3,32,32)
        self.assertEqual(model.forward_features(images).shape, (4,64))
        before = model.conv1.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
        loss = nn.functional.cross_entropy(model(images),torch.tensor([0,1,2,3]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertGreater(float(torch.linalg.vector_norm(model.conv1.weight.detach()-before)),0)

    def test_checkpoint_reload_and_freeze_preserve_logits_and_batchnorm(self):
        torch.manual_seed(7)
        model = ResNet32().eval()
        images = torch.randn(4,3,32,32)
        with torch.inference_mode():
            expected = model(images)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"weights.pt"
            torch.save(model.state_dict(),path)
            restored=ResNet32().eval()
            restored.load_state_dict(torch.load(path,weights_only=True))
        freeze_backbone(restored)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        self.assertTrue(all(not m.training for m in restored.modules()))
        before={name:value.clone() for name,value in restored.state_dict().items()}
        with torch.inference_mode():
            actual=restored(images)
            restored.forward_features(images)
        torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        for name,value in restored.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)

    def test_loader_test_split_contains_all_60_seen_classes_with_same_mapping(self):
        protocol = make_protocol(42, 5)
        images = np.zeros((100, 32, 32, 3), dtype=np.uint8)
        labels = np.arange(100)
        arrays = CifarArrays(images, labels, images, labels, tuple(map(str, labels)), {})
        config = {"seed": 42, "batch_size": 8, "eval_batch_size": 8}
        loader = make_loader(arrays, protocol, {"mean": [0.5]*3, "std": [0.25]*3},
                             range(60), "test", config)
        mapped = np.concatenate([y.numpy() for _, y in loader])
        np.testing.assert_array_equal(np.sort(mapped), np.arange(60))
        self.assertEqual(len(loader.dataset), 60)
        # Nhãn CIFAR 83 là class thứ 50 của permutation, không được đặt lại thành 0.
        self.assertEqual(protocol["original_to_global"][83], 50)

    def test_checkpoint_rejects_changed_mapping_preprocessing_or_training(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/cifar100_smoke.json").read_text(encoding="utf-8"))
        protocol = make_protocol(42, 5)
        norm = {"mean": [0.5]*3, "std": [0.25]*3}
        checkpoint = {"config": config, "class_order": protocol["class_order"],
                      "normalization": norm, "dataset_revision": "revision",
                      "training_history": [{"train_loss": 1.0}]*2, "parameter_change_l2": 1.0}
        validate_checkpoint(checkpoint, config, protocol, norm, "revision")
        changes = [lambda c: c["class_order"].reverse(),
                   lambda c: c["normalization"]["mean"].__setitem__(0, 0.1),
                   lambda c: c["config"].__setitem__("epochs", 3),
                   lambda c: c.__setitem__("dataset_revision", "other")]
        for change in changes:
            altered = copy.deepcopy(checkpoint)
            change(altered)
            with self.assertRaises(ValueError):
                validate_checkpoint(altered, config, protocol, norm, "revision")

    def test_feature_extraction_refuses_train_mode_or_unfrozen_backbone(self):
        model = ResNet32()
        batch = [(torch.zeros(2, 3, 32, 32), torch.tensor([0, 1]))]
        with self.assertRaises(ValueError):
            extract_features(model, batch, "cpu", "test")
        model.eval()
        with self.assertRaises(ValueError):
            extract_features(model, batch, "cpu", "test")
        freeze_backbone(model)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        features, labels = extract_features(model, batch, "cpu", "test")
        self.assertEqual(features.shape, (2, 64))
        self.assertEqual(features.dtype, np.float64)
        np.testing.assert_array_equal(labels, [0, 1])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

import h5py
import numpy as np

with h5py.File("samples.h5", "r") as f:
    print("Keys:")
    def print_key(name):
        print(name)
    f.visit(print_key)

    gt = f["expmap/gt/walking_0"][:]
    pred = f["expmap/preds/walking_0"][:]

print("GT shape:", gt.shape)
print("Pred shape:", pred.shape)

print("\nGT frame-to-frame mean movement:")
print(np.mean(np.linalg.norm(gt[1:] - gt[:-1], axis=1)))

print("\nPred frame-to-frame mean movement:")
print(np.mean(np.linalg.norm(pred[1:] - pred[:-1], axis=1)))

print("\nPred first-last difference:")
print(np.linalg.norm(pred[-1] - pred[0]))

print("\nFirst 5 pred frame diffs:")
print(np.linalg.norm(pred[1:6] - pred[:5], axis=1))
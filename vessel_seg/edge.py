"""Deep edge detection network and CLI utilities."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError("PyTorch is required for vessel_seg.edge. Install with `pip install torch`.") from exc


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _load_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            key = arr.files[0]
            arr = arr[key]
        return np.asarray(arr, dtype=np.float32)
    image = nib.load(str(path))
    return image.get_fdata().astype(np.float32)


def _window_intensity(slice_arr: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    low, high = window
    clipped = np.clip(slice_arr, low, high)
    scaled = (clipped - low) / max(high - low, 1e-3)
    return scaled.astype(np.float32)


def _iter_slice_indices(shape: Tuple[int, ...], axis: int) -> Iterable[Tuple[int, ...]]:
    for idx in range(shape[axis]):
        yield idx


@dataclass
class SliceRecord:
    image_path: Path
    edge_path: Path
    axis: int = 2


class EdgeSliceDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SliceRecord],
        window: Tuple[float, float] = (-200.0, 800.0),
    ) -> None:
        self.records = list(records)
        self.window = window
        self.samples: List[Tuple[Path, Path, int, int]] = []
        for record_idx, record in enumerate(self.records):
            image = _load_volume(record.image_path)
            edge = _load_volume(record.edge_path)
            if image.shape != edge.shape:
                raise ValueError(f"Shape mismatch for {record.image_path} and {record.edge_path}.")
            axis = record.axis
            for slice_idx in _iter_slice_indices(image.shape, axis):
                self.samples.append((record.image_path, record.edge_path, axis, slice_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, edge_path, axis, slice_idx = self.samples[index]
        volume = _load_volume(img_path)
        edge = _load_volume(edge_path)
        image_slice = np.take(volume, slice_idx, axis=axis)
        edge_slice = np.take(edge, slice_idx, axis=axis)
        image_slice = _window_intensity(image_slice, self.window)
        edge_slice = (edge_slice > 0.5).astype(np.float32)
        image_slice = np.expand_dims(image_slice, axis=0)
        edge_slice = np.expand_dims(edge_slice, axis=0)
        return torch.from_numpy(image_slice), torch.from_numpy(edge_slice)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        return self.block(x)


class EdgeFeatureNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up0 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(base_channels * 2, base_channels)

        self.classifier = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - thin wrapper
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.bottleneck(self.pool3(e3))

        d2 = self.up2(bottleneck)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d0 = self.up0(d1)
        d0 = torch.cat([d0, e1], dim=1)
        d0 = self.dec0(d0)

        return self.classifier(d0)


# --------------------------------------------------------------------------- #
# Training and inference
# --------------------------------------------------------------------------- #


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(pred)
    intersect = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * intersect + eps) / (denom + eps)


def train_model(
    manifest_path: str,
    output_weights: str,
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    window: Tuple[float, float] = (-200.0, 800.0),
    device: str = "cuda",
) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    records = [
        SliceRecord(image_path=Path(entry["image"]), edge_path=Path(entry["edge"]), axis=entry.get("axis", 2))
        for entry in manifest
    ]
    dataset = EdgeSliceDataset(records, window=window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = EdgeFeatureNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, edges in loader:
            images = images.to(device)
            edges = edges.to(device)
            logits = model(images)
            loss = criterion(logits, edges) + _dice_loss(logits, edges)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    torch.save(model.state_dict(), output_weights)
    print(f"Saved edge detector weights to {output_weights}")


def predict_volume(
    weights_path: str,
    volume_path: str,
    output_path: str,
    window: Tuple[float, float] = (-200.0, 800.0),
    axis: int = 2,
    device: str = "cuda",
) -> None:
    model = EdgeFeatureNet().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    volume = _load_volume(Path(volume_path))
    slices: List[np.ndarray] = []
    with torch.no_grad():
        for idx in _iter_slice_indices(volume.shape, axis):
            slice_arr = np.take(volume, idx, axis=axis)
            slice_arr = _window_intensity(slice_arr, window)
            tensor = torch.from_numpy(slice_arr[None, None, ...]).to(device)
            logits = model(tensor)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            slices.append(prob.astype(np.float32))
    stacked = np.stack(slices, axis=axis)

    reference = nib.load(volume_path)
    edge_img = nib.Nifti1Image(stacked.astype(np.float32), reference.affine, reference.header)
    nib.save(edge_img, output_path)
    print(f"Saved edge confidence map to {output_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deep edge detection utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the edge detector from a manifest.")
    train_parser.add_argument("--manifest", required=True, help="JSON manifest with image/edge pairs.")
    train_parser.add_argument("--output", required=True, help="Path to save trained weights (.pth).")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=(-200.0, 800.0),
        metavar=("LOW", "HIGH"),
        help="HU window for slice normalisation.",
    )
    train_parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    train_parser.set_defaults(func=lambda args: train_model(
        args.manifest,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        window=tuple(args.window),
        device=args.device,
    ))

    predict_parser = subparsers.add_parser("predict", help="Generate edge probabilities for a volume.")
    predict_parser.add_argument("--weights", required=True, help="Path to trained weights (.pth).")
    predict_parser.add_argument("--volume", required=True, help="Input volume (NIfTI/NPY).")
    predict_parser.add_argument("--output", required=True, help="Output NIfTI with edge probabilities.")
    predict_parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=(-200.0, 800.0),
        metavar=("LOW", "HIGH"),
    )
    predict_parser.add_argument("--axis", type=int, default=2, choices=(0, 1, 2))
    predict_parser.add_argument("--device", default="cuda")
    predict_parser.set_defaults(func=lambda args: predict_volume(
        args.weights,
        args.volume,
        args.output,
        window=tuple(args.window),
        axis=args.axis,
        device=args.device,
    ))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

#!/usr/bin/env python3
"""Teacher/Student dynamic-thinking distillation on full vs partial features."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def unpack_edge_density(edge_packed: np.ndarray, edge_shape: np.ndarray) -> float:
    if edge_shape.size == 0 or edge_shape[0] == 0:
        return 0.0
    h, w = int(edge_shape[0]), int(edge_shape[1])
    bits = np.unpackbits(edge_packed)[: h * w]
    return float(bits.mean())


def depth_stats(depth_u16: np.ndarray) -> np.ndarray:
    d = depth_u16.astype(np.float32) / 65535.0
    return np.array([d.mean(), d.std(), d.min(), d.max()], dtype=np.float32)


def bboxes_stats(bboxes: np.ndarray) -> np.ndarray:
    if bboxes.size == 0:
        return np.zeros((5,), dtype=np.float32)
    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    return np.array(
        [len(bboxes), bboxes[:, 0].mean(), bboxes[:, 1].mean(), bboxes[:, 2].mean(), area.mean()],
        dtype=np.float32,
    )


def vectorize_npz(npz_path: str) -> np.ndarray:
    x = np.load(npz_path, allow_pickle=True)
    bb = bboxes_stats(x["bboxes"])
    ds = depth_stats(x["depth_u16"])
    ed = np.array([unpack_edge_density(x["edge_packed"], x["edge_shape"])], dtype=np.float32)
    return np.concatenate([bb, ds, ed], axis=0).astype(np.float32)


class DistillDataset(Dataset):
    def __init__(self, manifest_path: str, limit: int = 1000):
        self.rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
                if len(self.rows) >= limit:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        full = vectorize_npz(r["full_npz_path"])
        part = vectorize_npz(r["partial_npz_path"])
        action = np.asarray(r["action"], dtype=np.float32)
        return torch.from_numpy(full), torch.from_numpy(part), torch.from_numpy(action)


class PolicyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--teacher_hidden", type=int, default=1024)
    parser.add_argument("--student_hidden", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = DistillDataset(args.dropout_manifest, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    in_dim = len(vectorize_npz(ds.rows[0]["full_npz_path"]))
    teacher = PolicyMLP(in_dim, hidden=args.teacher_hidden).to(device)
    student = PolicyMLP(in_dim, hidden=args.student_hidden).to(device)
    opt_t = torch.optim.AdamW(teacher.parameters(), lr=args.lr)
    opt_s = torch.optim.AdamW(student.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.jsonl"

    with metrics_path.open("w", encoding="utf-8") as mf:
        for ep in range(1, args.epochs + 1):
            teacher.train()
            student.train()
            sum_t, sum_s, steps = 0.0, 0.0, 0
            epoch_start = time.time()
            total_steps = len(dl)
            print(f"[epoch] {ep}/{args.epochs} steps={total_steps}", flush=True)
            for full_x, part_x, action in dl:
                full_x = full_x.to(device)
                part_x = part_x.to(device)
                action = action.to(device)

                t_pred = teacher(full_x)
                t_loss = mse(t_pred, action)
                opt_t.zero_grad()
                t_loss.backward()
                opt_t.step()

                with torch.no_grad():
                    t_target = teacher(full_x)
                s_pred = student(part_x)
                s_action = mse(s_pred, action)
                s_cons = mse(s_pred, t_target)
                s_loss = s_action + args.consistency_weight * s_cons

                opt_s.zero_grad()
                s_loss.backward()
                opt_s.step()

                sum_t += float(t_loss.item())
                sum_s += float(s_loss.item())
                steps += 1

                if steps % args.log_every == 0 or steps == total_steps:
                    elapsed = max(1e-6, time.time() - epoch_start)
                    speed = steps / elapsed
                    remain = max(0, total_steps - steps)
                    eta_sec = int(remain / max(1e-6, speed))
                    print(
                        f"[progress] epoch={ep}/{args.epochs} step={steps}/{total_steps} "
                        f"teacher_loss={t_loss.item():.6f} student_loss={s_loss.item():.6f} "
                        f"speed={speed:.2f} steps/s eta={eta_sec}s",
                        flush=True,
                    )

            rec = {
                "epoch": ep,
                "teacher_loss": sum_t / max(1, steps),
                "student_loss": sum_s / max(1, steps),
                "consistency_weight": args.consistency_weight,
                "teacher_hidden": args.teacher_hidden,
                "student_hidden": args.student_hidden,
            }
            mf.write(json.dumps(rec) + "\n")
            print(rec, flush=True)

    torch.save(teacher.state_dict(), out_dir / "teacher.pt")
    torch.save(student.state_dict(), out_dir / "student.pt")
    print(f"[ok] saved: {out_dir}")


if __name__ == "__main__":
    main()

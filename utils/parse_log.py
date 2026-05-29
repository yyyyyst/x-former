#!/usr/bin/env python3
"""从日志文件提取实验结果。用法: python3 utils/parse_log.py logs/xformer_*.log"""
import sys
import re

def parse_log(path):
    with open(path) as f:
        text = f.read()

    folds = []
    for m in re.finditer(
        r"Fold (\d+) best val AUC: ([0-9.]+).+?77sets test AUC: ([0-9.]+)\s+ISLE test AUC: ([0-9.]+)",
        text, re.DOTALL
    ):
        folds.append({
            "fold": int(m.group(1)),
            "val_auc": float(m.group(2)),
            "auc_77": float(m.group(3)),
            "auc_isle": float(m.group(4)),
        })

    final = {}
    m = re.search(r"77sets:\s+auc: ([0-9.]+) \+/- ([0-9.]+).+?ISLE2024:\s+auc: ([0-9.]+) \+/- ([0-9.]+)", text, re.DOTALL)
    if m:
        final["77_mean"], final["77_std"] = float(m.group(1)), float(m.group(2))
        final["isle_mean"], final["isle_std"] = float(m.group(3)), float(m.group(4))

    pooled = {}
    for ds in ["77sets", "ISLE2024"]:
        m = re.search(rf"{ds} \(pooled\):.+?auc: ([0-9.]+) \(95% CI: ([0-9.]+)-([0-9.]+)\)", text, re.DOTALL)
        if m:
            pooled[ds] = {"auc": float(m.group(1)), "ci_lower": float(m.group(2)), "ci_upper": float(m.group(3))}

    return folds, final, pooled


if __name__ == "__main__":
    for path in sys.argv[1:]:
        folds, final, pooled = parse_log(path)
        print(f"\n{'='*60}")
        print(f"文件: {path}")
        print(f"{'='*60}")
        print(f"折数: {len(folds)}")
        print(f"\n{'Fold':<6} {'Val AUC':<10} {'77sets':<10} {'ISLE':<10}")
        print("-" * 36)
        for f in folds:
            print(f"  {f['fold']:<4} {f['val_auc']:<10.4f} {f['auc_77']:<10.4f} {f['auc_isle']:<10.4f}")
        if final:
            print(f"\n逐折均值:  77sets={final['77_mean']:.4f}±{final['77_std']:.4f}  ISLE={final['isle_mean']:.4f}±{final['isle_std']:.4f}")
        if pooled:
            for ds, p in pooled.items():
                print(f"{ds} pooled: AUC={p['auc']:.4f} (95% CI: {p['ci_lower']:.4f}-{p['ci_upper']:.4f})")

"""Create non-destructive prediction/annotation diagnostics for a saved checkpoint."""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np, torch
from .dataset import discover_pairs, split_pairs
from .models import build_model
from .preprocessing import prepare
from .utils import metrics


def _panel(image, truth, prob, pred):
    h, w = truth.shape
    original = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    gt = cv2.cvtColor((truth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    pm = cv2.applyColorMap((prob * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    pp = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay = original.copy(); overlay[pred] = (0, 0, 255); overlay[truth] = (0, 255, 0)
    err = np.zeros_like(original); err[(pred & ~truth)] = (0, 0, 255); err[(truth & ~pred)] = (255, 0, 0)
    return np.concatenate([original, gt, pm, pp, overlay, err], axis=1)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path); p.add_argument('--device', default='cpu'); p.add_argument('--threshold', type=float, default=.5); p.add_argument('--seed', type=int, default=42); p.add_argument('--splits', nargs='+', default=['train','val','test']); a = p.parse_args()
    bundle = torch.load(a.checkpoint, map_location='cpu', weights_only=False); cfg = bundle.get('config', {})
    model = build_model(bundle['architecture'], cfg.get('encoder', 'efficientnet-b0'), False); model.load_state_dict(bundle['state_dict']); model.to(a.device).eval()
    pairs, _ = discover_pairs(a.data); train, val, test = split_pairs(pairs, cfg.get('val_fraction', .15), cfg.get('test_fraction', .15), cfg.get('seed', a.seed))
    split = {str(x.image): name for name, xs in [('train', train), ('val', val), ('test', test)] for x in xs}; pairs = [x for x in pairs if split[str(x.image)] in set(a.splits)]
    out = a.output or a.checkpoint.parent / 'diagnostics'; out.mkdir(parents=True, exist_ok=True); overlays = out / 'overlays'; overlays.mkdir(exist_ok=True)
    rows = []; probabilities = {}; masks = {}; images = {}; batch_size = 8
    prepared = []
    for pair in pairs:
        image = cv2.cvtColor(cv2.imread(str(pair.image)), cv2.COLOR_BGR2RGB); mask = cv2.imread(str(pair.mask), cv2.IMREAD_GRAYSCALE)
        x, _ = prepare(pair.image, pair.mask, int(cfg.get('size', 320)), False); prepared.append((pair, image, mask, x))
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]; logits = model(torch.stack([x for _, _, _, x in chunk]).to(a.device)); small = torch.sigmoid(logits).cpu().numpy()[:, 0]
            for (pair, image, mask, _), prob_small in zip(chunk, small):
                h, w = mask.shape; prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR); truth = mask > 0; pred = prob >= a.threshold
                z = metrics(torch.from_numpy(np.log(np.clip(prob, 1e-6, 1-1e-6) / np.clip(1-prob, 1e-6, 1))), torch.from_numpy(truth.astype(np.float32))[None, None], a.threshold)
                key = str(pair.image); probabilities[key] = prob; masks[key] = truth; images[key] = image
                row = {'image': key, 'split': split[key], 'gt_fraction': float(truth.mean()), 'pred_fraction': float(pred.mean()), 'prob_min': float(prob.min()), 'prob_max': float(prob.max()), 'prob_mean': float(prob.mean()), 'prob_p10': float(np.percentile(prob, 10)), 'prob_p50': float(np.percentile(prob, 50)), 'prob_p90': float(np.percentile(prob, 90)), **z}; rows.append(row)
    with (out / 'per_image.csv').open('w', newline='', encoding='utf-8') as f: w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for split_name in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split_name]; subset.sort(key=lambda r: r['dice'])
        chosen = subset[:5] + subset[-3:]
        for r in chosen:
            pair = next(x for x in pairs if str(x.image) == r['image']); image = images[r['image']]; truth = masks[r['image']]; prob = probabilities[r['image']]; panel = _panel(image, truth, prob, prob >= a.threshold); cv2.imwrite(str(overlays / f"{split_name}_{Path(pair.image).stem}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    summary = {s: {'count': len([r for r in rows if r['split'] == s]), 'mean_dice': float(np.mean([r['dice'] for r in rows if r['split'] == s])), 'mean_gt_fraction': float(np.mean([r['gt_fraction'] for r in rows if r['split'] == s])), 'mean_pred_fraction': float(np.mean([r['pred_fraction'] for r in rows if r['split'] == s])), 'mean_prob': float(np.mean([r['prob_mean'] for r in rows if r['split'] == s]))} for s in ('train', 'val', 'test')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8'); print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()

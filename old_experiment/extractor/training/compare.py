"""Compare recorded validation/test summaries without retraining."""
import argparse,json,csv
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('extractor/experiments')); a=p.parse_args(); rows=[]
    for d in a.root.iterdir() if a.root.is_dir() else []:
        history=d/'history.json'; checkpoint=d/'best_checkpoint.pt'
        if not history.is_file(): continue
        entries=json.loads(history.read_text()); best=max(entries,key=lambda x:x.get('val_dice',x.get('dice',-1)))
        import torch
        bundle=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.is_file() else {}
        rows.append({'architecture':d.name,'best_epoch':best.get('epoch'),'val_dice':best.get('dice',best.get('val_dice')),'val_iou':best.get('iou',best.get('val_iou')),'val_loss':best.get('val_loss'),'parameters':sum(v.numel() for v in bundle.get('state_dict',{}).values()),'checkpoint_bytes':checkpoint.stat().st_size if checkpoint.is_file() else 0})
    if not rows: print(f'No completed experiments under {a.root}; train all three models first.'); return
    rows=sorted(rows,key=lambda x:x.get('val_dice',-1),reverse=True)
    for r in rows: print(r)
    comparison=a.root/'comparison'; comparison.mkdir(parents=True,exist_ok=True)
    with (comparison/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (comparison/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (comparison/'report.md').write_text('# Segmentation comparison\n\nThe winner is selected by validation Dice only; the held-out test set is not used for selection.\n\n'+ '\n'.join(f"- **{r['architecture']}**: validation Dice {r['val_dice']:.6f}, IoU {r['val_iou']:.6f}, parameters {r['parameters']:,}" for r in rows)+'\n',encoding='utf-8')
if __name__=='__main__': main()

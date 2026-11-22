#!/usr/bin/env python3
"""Manual conjunctiva-mask annotation. New masks start BLANK; no model is loaded."""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path
import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
WINDOW = "Manual conjunctiva annotation"

def load_manifest(path):
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}

def save_manifest(path, records):
    fields = ["image_id", "image_file", "mask_file", "status",
              "verified_by", "verified_at", "notes"]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(records):
            w.writerow({k: records[key].get(k, "") for k in fields})
    tmp.replace(path)

def fit_geometry(image, max_w, max_h):
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    return scale, (max_w-dw)//2, (max_h-dh)//2, dw, dh

def display_to_image(x, y, geom, shape):
    scale, ox, oy, dw, dh = geom
    if not (ox <= x < ox+dw and oy <= y < oy+dh):
        return None
    h, w = shape[:2]
    return (min(w-1, max(0, int((x-ox)/scale))),
            min(h-1, max(0, int((y-oy)/scale))))

class Annotator:
    def __init__(self, images_dir, output_dir, reviewer, brush):
        self.images_dir, self.output_dir = images_dir, output_dir
        self.draft_dir = output_dir / "masks" / "drafts"
        self.verified_dir = output_dir / "masks" / "verified"
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_dir / "reports" / "annotation_manifest.csv"
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = load_manifest(self.manifest_path)
        self.reviewer, self.brush = reviewer, max(1, brush)
        self.files = sorted(p for p in images_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in EXTS)
        if not self.files:
            raise RuntimeError(f"No images found in {images_dir}")
        self.index = 0
        self.width, self.height = 1200, 850
        self.overlay, self.dirty = True, False
        self.drawing, self.erase, self.last = False, False, None
        self.load_image()

    def load_image(self):
        self.path = self.files[self.index]
        self.image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Cannot read {self.path}")
        h, w = self.image.shape[:2]
        record = self.records.get(self.path.stem, {})
        recorded_mask = record.get("mask_file", "")
        self.mask_path = self.output_dir / recorded_mask if recorded_mask else self.draft_dir / f"{self.path.stem}.png"
        if self.mask_path.exists():
            old = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if old is None or old.shape != (h, w):
                raise RuntimeError(f"Existing mask dimensions invalid: {self.mask_path}")
            self.mask = np.where(old > 127, 255, 0).astype(np.uint8)
        else:
            # Crucial: start from an empty mask, never an old model prediction.
            self.mask = np.zeros((h, w), dtype=np.uint8)
        self.geom = fit_geometry(self.image, self.width, self.height-90)
        self.dirty, self.last = False, None
        iid = self.path.stem
        if iid not in self.records:
            self.records[iid] = dict(image_id=iid, image_file=self.path.name,
                mask_file=f"masks/drafts/{self.path.stem}.png", status="pending",
                verified_by="", verified_at="", notes="")
            save_manifest(self.manifest_path, self.records)

    def mouse(self, event, x, y, flags, _):
        pt = display_to_image(x, y, self.geom, self.image.shape)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and pt:
            self.drawing = True
            self.erase = event == cv2.EVENT_RBUTTONDOWN
            self.last = pt
            cv2.circle(self.mask, pt, self.brush, 0 if self.erase else 255, -1)
            self.dirty = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and pt:
            value = 0 if self.erase else 255
            cv2.line(self.mask, self.last, pt, value, self.brush*2)
            cv2.circle(self.mask, pt, self.brush, value, -1)
            self.last, self.dirty = pt, True
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing, self.last = False, None

    def save(self, verified):
        iid = self.path.stem
        binary = np.where(self.mask > 0, 255, 0).astype(np.uint8)
        self.mask_path = (self.verified_dir if verified else self.draft_dir) / f"{self.path.stem}.png"
        if not cv2.imwrite(str(self.mask_path), binary):
            raise RuntimeError(f"Could not save {self.mask_path}")
        row = self.records[iid]
        row["mask_file"] = str(self.mask_path.relative_to(self.output_dir)).replace("\\", "/")
        row["status"] = "verified" if verified else "draft"
        row["verified_by"] = self.reviewer if verified else ""
        row["verified_at"] = time.strftime("%Y-%m-%d %H:%M:%S") if verified else ""
        save_manifest(self.manifest_path, self.records)
        self.dirty = False
        print(f"{'VERIFIED' if verified else 'Draft saved'}: {self.path.name}")

    def next(self, step):
        if self.dirty:
            print("Unsaved changes. Press S to save draft or V to verify first.")
            return
        target = self.index + step
        if 0 <= target < len(self.files):
            self.index = target
            self.load_image()

    def render(self):
        canvas = np.zeros((self.height, self.width, 3), np.uint8)
        scale, ox, oy, dw, dh = self.geom
        shown = cv2.resize(self.image, (dw, dh), interpolation=cv2.INTER_AREA)
        if self.overlay:
            m = cv2.resize(self.mask, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            shown[m] = (0.55*shown[m] + 0.45*np.array([0, 0, 255])).astype(np.uint8)
        canvas[oy:oy+dh, ox:ox+dw] = shown
        row = self.records[self.path.stem]
        verified = sum(r.get("status") == "verified" for r in self.records.values())
        cv2.rectangle(canvas, (0, self.height-90), (self.width, self.height), (25,25,25), -1)
        title = (f"{self.index+1}/{len(self.files)} {self.path.name} | "
                 f"{row.get('status','pending')} | Verified {verified}/{len(self.files)} | "
                 f"Brush radius {self.brush}")
        cv2.putText(canvas, title[:140], (12,self.height-58),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
        helptext = "Left draw | Right erase | [ ] brush | O overlay | C clear | S draft+next | V verify+next | P prev | N next | Q quit"
        cv2.putText(canvas, helptext, (12,self.height-23),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, (180,220,255), 1, cv2.LINE_AA)
        if self.dirty:
            cv2.putText(canvas, "UNSAVED", (self.width-170, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
        return canvas

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.width, self.height)
        cv2.setMouseCallback(WINDOW, self.mouse)
        print("L-draw, R-erase, S-save draft, V-verify, P/N-navigate, Q-quit")
        while True:
            cv2.imshow(WINDOW, self.render())
            k = cv2.waitKey(20) & 0xff
            if k in (ord("q"), 27):
                break
            elif k == ord("o"):
                self.overlay = not self.overlay
            elif k == ord("["):
                self.brush = max(1, self.brush-2)
            elif k == ord("]"):
                self.brush = min(200, self.brush+2)
            elif k == ord("c"):
                self.mask[:] = 0
                self.dirty = True
            elif k == ord("s"):
                self.save(False); self.next(1)
            elif k == ord("v"):
                if np.count_nonzero(self.mask) == 0:
                    print("Cannot verify an empty mask.")
                else:
                    self.save(True); self.next(1)
            elif k == ord("n"):
                self.next(1)
            elif k == ord("p"):
                self.next(-1)
        cv2.destroyAllWindows()
        print(f"Manifest: {self.manifest_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reviewer", default="self")
    p.add_argument("--brush-size", type=int, default=18)
    a = p.parse_args()
    if not a.images.is_dir():
        p.error(f"Images folder not found: {a.images}")
    a.output.mkdir(parents=True, exist_ok=True)
    Annotator(a.images, a.output, a.reviewer, a.brush_size).run()

if __name__ == "__main__":
    main()

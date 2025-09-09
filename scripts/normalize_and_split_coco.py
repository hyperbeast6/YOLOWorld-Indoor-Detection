import argparse
import json
import os
import random
from collections import defaultdict


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_filenames(coco: dict, prefix: str | None = 'images/', use_basename: bool = True, use_posix_slash: bool = True):
    for img in coco.get('images', []):
        file_name = img.get('file_name', '')
        if use_basename:
            file_name = os.path.basename(file_name)
        if use_posix_slash:
            file_name = file_name.replace('\\', '/')
        if prefix:
            # 避免重复前缀
            if not file_name.startswith(prefix):
                file_name = f"{prefix}{file_name}"
        img['file_name'] = file_name


def validate_and_clip_bboxes(coco: dict, clip: bool = True) -> tuple[int, int]:
    """返回 (修正数量, 丢弃数量)"""
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco.get('images', [])}
    fixed = 0
    dropped = 0
    new_anns = []
    for ann in coco.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id not in image_id_to_size:
            dropped += 1
            continue
        w, h = image_id_to_size[img_id]
        x, y, bw, bh = ann.get('bbox', [0, 0, 0, 0])
        # 负值纠正
        if bw < 0 or bh < 0:
            dropped += 1
            continue
        if clip:
            nx = max(0.0, min(float(x), float(w)))
            ny = max(0.0, min(float(y), float(h)))
            nxe = max(0.0, min(float(x + bw), float(w)))
            nye = max(0.0, min(float(y + bh), float(h)))
            nbw = max(0.0, nxe - nx)
            nbh = max(0.0, nye - ny)
            if (nx != x) or (ny != y) or (abs(nbw - bw) > 1e-6) or (abs(nbh - bh) > 1e-6):
                fixed += 1
            ann['bbox'] = [round(nx, 2), round(ny, 2), round(nbw, 2), round(nbh, 2)]
            ann['area'] = round(nbw * nbh, 4)
            # 丢弃零面积框
            if nbw <= 0 or nbh <= 0:
                dropped += 1
                continue
        new_anns.append(ann)
    coco['annotations'] = new_anns
    return fixed, dropped


def reindex_annotations(coco: dict):
    for new_id, ann in enumerate(coco.get('annotations', [])):
        ann['id'] = new_id


def ensure_contiguous_category_ids(coco: dict):
    # 将 categories 按 id 排序，并确保从 0..N-1 连续（如不连续则重映射）
    cats = coco.get('categories', [])
    cats_sorted = sorted(cats, key=lambda c: c['id'])
    id_map = {}
    for new_id, cat in enumerate(cats_sorted):
        id_map[cat['id']] = new_id
        cat['id'] = new_id
    coco['categories'] = cats_sorted
    # 重映射 annotations 的 category_id
    for ann in coco.get('annotations', []):
        old = ann['category_id']
        if old in id_map:
            ann['category_id'] = id_map[old]


def split_coco(coco: dict, train_ratio: float, seed: int = 42):
    assert 0.0 < train_ratio < 1.0
    rng = random.Random(seed)
    image_ids = [img['id'] for img in coco.get('images', [])]
    rng.shuffle(image_ids)
    split_idx = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    def subset(ids: set[int]):
        images = [img for img in coco['images'] if img['id'] in ids]
        id_set = set(ids)
        anns = [ann for ann in coco['annotations'] if ann['image_id'] in id_set]
        sub = {
            'images': images,
            'annotations': anns,
            'categories': coco.get('categories', []),
            'info': coco.get('info', {})
        }
        return sub

    train_coco = subset(train_ids)
    val_coco = subset(val_ids)
    reindex_annotations(train_coco)
    reindex_annotations(val_coco)
    return train_coco, val_coco


def main():
    parser = argparse.ArgumentParser(description='Normalize COCO JSON from Label Studio and split train/val')
    parser.add_argument('--input', required=True, help='Path to source COCO JSON (e.g., data/indoor_dataset_cocostyle/result.json)')
    parser.add_argument('--output-dir', required=True, help='Directory to write instances_train.json and instances_val.json')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    parser.add_argument('--no-prefix', action='store_true', help='Do not prepend images/ prefix in file_name')
    parser.add_argument('--keep-absolute', action='store_true', help='Keep absolute file paths (not recommended)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for split')
    args = parser.parse_args()

    coco = load_json(args.input)

    prefix = None if args.no_prefix else 'images/'
    use_basename = not args.keep_absolute
    normalize_filenames(coco, prefix=prefix, use_basename=use_basename, use_posix_slash=True)

    ensure_contiguous_category_ids(coco)
    fixed, dropped = validate_and_clip_bboxes(coco, clip=True)

    print(f"Fixed {fixed} bboxes; Dropped {dropped} invalid annotations.")

    train_coco, val_coco = split_coco(coco, train_ratio=args.train_ratio, seed=args.seed)

    train_path = os.path.join(args.output_dir, 'instances_train.json')
    val_path = os.path.join(args.output_dir, 'instances_val.json')
    save_json(train_coco, train_path)
    save_json(val_coco, val_path)
    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")


if __name__ == '__main__':
    main()



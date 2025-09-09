import json, os
src="data/labelstudio_import/labelstudio_annotations.json"
dst="data/labelstudio_import/labelstudio_annotations_fixed.json"
with open(src, "r", encoding="utf-8") as f:
    items=json.load(f)
fixed=[]
for it in items:
    img_path=it["data"]["image"]
    meta=it.get("meta", {})
    ow, oh = meta.get("width"), meta.get("height")
    results=[]
    # 旧文件里把每个框直接放在 annotations 数组里
    for ann in it.get("annotations", []):
        v=ann["value"]
        results.append({
            "id": str(ann.get("id", "")),
            "from_name": "objects",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": v["x"], "y": v["y"], "width": v["width"], "height": v["height"],
                "rotation": v.get("rotation", 0),
                "rectanglelabels": v["rectanglelabels"]
            },
            "original_width": ow, "original_height": oh, "image_rotation": 0
        })
    fixed.append({
        "data": {"image": img_path},
        "annotations": [{"result": results, "ground_truth": False}]
    })
with open(dst, "w", encoding="utf-8") as f:
    json.dump(fixed, f, ensure_ascii=False, indent=2)
print("OK ->", dst)
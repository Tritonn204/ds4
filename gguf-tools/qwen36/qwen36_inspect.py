#!/usr/bin/env python3

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict


GGUF_MAGIC = 0x46554747
GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

TYPE_NAMES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    24: "i8",
    25: "i16",
    26: "i32",
    27: "i64",
    28: "f64",
    29: "iq1_m",
    30: "bf16",
}

SCALAR_FORMATS = {
    GGUF_VALUE_UINT8: "<B",
    GGUF_VALUE_INT8: "<b",
    GGUF_VALUE_UINT16: "<H",
    GGUF_VALUE_INT16: "<h",
    GGUF_VALUE_UINT32: "<I",
    GGUF_VALUE_INT32: "<i",
    GGUF_VALUE_FLOAT32: "<f",
    GGUF_VALUE_BOOL: "<?",
    GGUF_VALUE_UINT64: "<Q",
    GGUF_VALUE_INT64: "<q",
    GGUF_VALUE_FLOAT64: "<d",
}

MODEL_PATTERNS = (
    "qwen3.6-27b",
    "qwen3.6-35b-a3b",
)

LAYER_PATTERNS = [
    re.compile(r"^blk\.(\d+)\.(.+)$"),
    re.compile(r"^model\.layers\.(\d+)\.(.+)$"),
    re.compile(r"^layers\.(\d+)\.(.+)$"),
    re.compile(r"^transformer\.h\.(\d+)\.(.+)$"),
]


class Cursor:
    def __init__(self, fh):
        self.fh = fh

    def read(self, n):
        out = self.fh.read(n)
        if len(out) != n:
            raise ValueError("unexpected EOF")
        return out

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def u32(self):
        return self.unpack("<I")

    def u64(self):
        return self.unpack("<Q")

    def string(self):
        n = self.u64()
        raw = self.read(n)
        return raw.decode("utf-8", errors="replace")

    def skip(self, n):
        self.fh.seek(n, os.SEEK_CUR)


def read_scalar(cur, value_type):
    fmt = SCALAR_FORMATS.get(value_type)
    if not fmt:
        raise ValueError(f"unsupported scalar metadata type {value_type}")
    return cur.unpack(fmt)


def scalar_size(value_type):
    fmt = SCALAR_FORMATS.get(value_type)
    if not fmt:
        return None
    return struct.calcsize(fmt)


def skip_string(cur):
    cur.skip(cur.u64())


def skip_value(cur, value_type):
    if value_type == GGUF_VALUE_STRING:
        skip_string(cur)
        return
    if value_type == GGUF_VALUE_ARRAY:
        item_type = cur.u32()
        length = cur.u64()
        item_size = scalar_size(item_type)
        if item_size is not None:
            cur.skip(item_size * length)
            return
        for _ in range(length):
            skip_value(cur, item_type)
        return
    size = scalar_size(value_type)
    if size is None:
        raise ValueError(f"unsupported scalar metadata type {value_type}")
    cur.skip(size)


def read_value(cur, value_type, max_array_items=64):
    if value_type == GGUF_VALUE_STRING:
        return cur.string()
    if value_type == GGUF_VALUE_ARRAY:
        item_type = cur.u32()
        length = cur.u64()
        kept = []
        clipped = False
        limit = min(length, max_array_items)
        for _ in range(limit):
            kept.append(read_value(cur, item_type, max_array_items=max_array_items))
        if length > limit:
            clipped = True
            for _ in range(length - limit):
                skip_value(cur, item_type)
        return {
            "item_type": item_type,
            "item_type_name": gguf_value_type_name(item_type),
            "length": length,
            "clipped": clipped,
            "value": kept,
        }
    return read_scalar(cur, value_type)


def gguf_value_type_name(value_type):
    names = {
        GGUF_VALUE_UINT8: "uint8",
        GGUF_VALUE_INT8: "int8",
        GGUF_VALUE_UINT16: "uint16",
        GGUF_VALUE_INT16: "int16",
        GGUF_VALUE_UINT32: "uint32",
        GGUF_VALUE_INT32: "int32",
        GGUF_VALUE_FLOAT32: "float32",
        GGUF_VALUE_BOOL: "bool",
        GGUF_VALUE_STRING: "string",
        GGUF_VALUE_ARRAY: "array",
        GGUF_VALUE_UINT64: "uint64",
        GGUF_VALUE_INT64: "int64",
        GGUF_VALUE_FLOAT64: "float64",
    }
    return names.get(value_type, f"unknown_{value_type}")


def parse_gguf(path):
    interesting_metadata_tags = (
        "general.",
        "qwen",
        "rope",
        "context",
        "block_count",
        "layer",
        "head",
        "expert",
        "router",
        "vision",
        "visual",
        "image",
        "mtp",
        "delta",
    )

    with open(path, "rb") as fh:
        cur = Cursor(fh)
        magic = cur.u32()
        if magic != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version = cur.u32()
        if version != 3:
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensors = cur.u64()
        n_kv = cur.u64()

        metadata = {}
        metadata_types = {}
        for _ in range(n_kv):
            key = cur.string()
            value_type = cur.u32()
            metadata_types[key] = gguf_value_type_name(value_type)
            lower = key.lower()
            if any(tag in lower for tag in interesting_metadata_tags):
                metadata[key] = read_value(cur, value_type)
            else:
                skip_value(cur, value_type)

        tensors = []
        for _ in range(n_tensors):
            name = cur.string()
            ndim = cur.u32()
            dims = [cur.u64() for _ in range(ndim)]
            tensor_type = cur.u32()
            rel_offset = cur.u64()
            tensors.append({
                "name": name,
                "ndim": ndim,
                "dims": dims,
                "type": tensor_type,
                "type_name": TYPE_NAMES.get(tensor_type, f"unknown_{tensor_type}"),
                "rel_offset": rel_offset,
            })

        file_size = os.fstat(fh.fileno()).st_size

    return {
        "path": path,
        "file_size": file_size,
        "gguf_version": version,
        "metadata": metadata,
        "metadata_types": metadata_types,
        "tensors": tensors,
    }


def normalize_model_text(text):
    return re.sub(r"[^a-z0-9.+-]+", "", text.lower())


def infer_target(metadata):
    candidates = []
    for key in ("general.name", "general.basename", "general.architecture"):
        value = metadata.get(key)
        if isinstance(value, str):
            candidates.append(value)
    joined = " ".join(candidates)
    normalized = normalize_model_text(joined)
    for pattern in MODEL_PATTERNS:
        if pattern in normalized:
            return pattern
    return None


def scalar_or_none(metadata, key):
    value = metadata.get(key)
    if isinstance(value, dict):
        return None
    return value


def family_for_name(name):
    lower = name.lower()
    for tag in (
        "attention", "attn", "mla", "moe", "router", "expert", "gate", "ffn",
        "deltanet", "delta", "vision", "visual", "image", "audio", "mtp",
        "embed", "embd", "token", "norm", "output", "lm_head", "rope", "kv",
    ):
        if tag in lower:
            return tag
    return lower.split(".", 1)[0]


def bucket_name(name):
    lower = name.lower()
    tags = []
    checks = [
        ("attention", ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo")),
        ("mla", ("mla", "latent", "lora_a", "lora_b")),
        ("ffn", ("mlp", "ffn", "up_proj", "down_proj", "gate_proj")),
        ("moe", ("moe", "expert", "experts", "router", "shared_expert", "sharedexperts", "_exps", "_shexp", "gate_inp")),
        ("ssm", ("ssm_", ".ssm", "state_space")),
        ("deltanet", ("delta", "deltanet")),
        ("state_cache", ("cache", "k_cache", "v_cache")),
        ("vision", ("vision", "visual", "image", "patch", "mmproj", "projector", "vision_tower")),
        ("mtp", ("mtp", "draft", "speculative")),
        ("embed", ("embed", "embd", "token_embd", "tok_embeddings")),
        ("norm", ("norm", "rms")),
        ("output", ("output", "lm_head", "logits", "score")),
    ]
    for tag, needles in checks:
        if any(needle in lower for needle in needles):
            tags.append(tag)
    return tags or ["unknown"]


def strip_layer_index(name):
    for pattern in LAYER_PATTERNS:
        match = pattern.match(name)
        if match:
            return int(match.group(1)), match.group(2)
    return None, name


def collect_layer_templates(tensors):
    templates = Counter()
    layers = defaultdict(list)
    detected = 0
    for tensor in tensors:
        layer_idx, suffix = strip_layer_index(tensor["name"])
        if layer_idx is None:
            continue
        detected += 1
        templates[suffix] += 1
        layers[layer_idx].append(suffix)
    return detected, templates, layers


def summarize_quant_surface(tensors):
    overall = Counter()
    by_bucket = defaultdict(Counter)
    for tensor in tensors:
        tname = tensor["type_name"]
        overall[tname] += 1
        for bucket in bucket_name(tensor["name"]):
            by_bucket[bucket][tname] += 1
    return {
        "overall": dict(sorted(overall.items())),
        "by_bucket": {k: dict(sorted(v.items())) for k, v in sorted(by_bucket.items())},
    }


def summarize_buckets(tensors):
    out = defaultdict(list)
    for tensor in tensors:
        info = {
            "name": tensor["name"],
            "dims": tensor["dims"],
            "type": tensor["type_name"],
        }
        for bucket in bucket_name(tensor["name"]):
            out[bucket].append(info)
    for key in out:
        out[key].sort(key=lambda item: item["name"])
    return out


def top_level_name_families(tensors):
    counter = Counter()
    for tensor in tensors:
        counter[family_for_name(tensor["name"])] += 1
    return dict(counter.most_common())


def extract_architecture_hints(metadata, tensors):
    moe_tensors = [t["name"] for t in tensors if any(k in t["name"].lower() for k in ("moe", "expert", "router", "shared_expert", "_exps", "_shexp", "gate_inp"))]
    vision_tensors = [t["name"] for t in tensors if any(k in t["name"].lower() for k in ("vision", "visual", "image", "patch", "mmproj", "projector"))]
    mtp_tensors = [t["name"] for t in tensors if any(k in t["name"].lower() for k in ("mtp", "draft", "speculative"))]
    deltanet_tensors = [t["name"] for t in tensors if "delta" in t["name"].lower()]
    attention_tensors = [t["name"] for t in tensors if any(k in t["name"].lower() for k in ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo"))]
    ssm_tensors = [t["name"] for t in tensors if any(k in t["name"].lower() for k in ("ssm_", ".ssm", "state_space"))]

    dense_or_moe = "moe" if moe_tensors else "dense_or_unknown"
    if scalar_or_none(metadata, "qwen35moe.expert_count") is not None:
        dense_or_moe = "moe"
    if scalar_or_none(metadata, "qwen3moe.expert_count") is not None:
        dense_or_moe = "moe"

    return {
        "dense_or_moe": dense_or_moe,
        "hybrid_attention_ssm": bool(attention_tensors) and bool(ssm_tensors),
        "has_attention_evidence": bool(attention_tensors),
        "has_moe_evidence": bool(moe_tensors),
        "has_ssm_evidence": bool(ssm_tensors),
        "has_deltanet_evidence": bool(deltanet_tensors),
        "has_vision_evidence": bool(vision_tensors),
        "has_mtp_evidence": bool(mtp_tensors),
        "example_tensors": {
            "attention": attention_tensors[:24],
            "moe": moe_tensors[:24],
            "ssm": ssm_tensors[:24],
            "deltanet": deltanet_tensors[:24],
            "vision": vision_tensors[:24],
            "mtp": mtp_tensors[:24],
        },
    }


def build_report(parsed):
    metadata = parsed["metadata"]
    tensors = parsed["tensors"]
    target = infer_target(metadata)
    if not target:
        raise ValueError("GGUF does not appear to be an exact Qwen3.6-27B or Qwen3.6-35B-A3B target")

    detected_layers, templates, layers = collect_layer_templates(tensors)
    bucketed = summarize_buckets(tensors)
    layer_counts = {str(idx): len(items) for idx, items in sorted(layers.items())}

    metadata_subset = {}
    for key in sorted(metadata):
        lower = key.lower()
        if any(tag in lower for tag in (
            "general.", "qwen", "rope", "context", "block_count", "layer", "head",
            "expert", "router", "vision", "visual", "image", "mtp", "delta"
        )):
            metadata_subset[key] = metadata[key]

    report = {
        "artifact": {
            "path": os.path.abspath(parsed["path"]),
            "file_size_bytes": parsed["file_size"],
            "gguf_version": parsed["gguf_version"],
            "target_model": target,
        },
        "metadata": metadata_subset,
        "tensor_inventory": {
            "tensor_count": len(tensors),
            "quant_histogram": summarize_quant_surface(tensors)["overall"],
            "top_level_name_families": top_level_name_families(tensors),
        },
        "layer_template": {
            "detected_layer_tensors": detected_layers,
            "detected_layer_count": len(layers),
            "repeated_suffixes": dict(templates.most_common(128)),
            "tensor_count_per_layer": layer_counts,
        },
        "attention_evidence": bucketed.get("attention", [])[:64],
        "moe_evidence": bucketed.get("moe", [])[:64],
        "ssm_evidence": bucketed.get("ssm", [])[:64],
        "deltanet_evidence": bucketed.get("deltanet", [])[:64],
        "state_cache_evidence": bucketed.get("state_cache", [])[:64],
        "extra_heads": {
            "vision": bucketed.get("vision", [])[:64],
            "mtp": bucketed.get("mtp", [])[:64],
        },
        "quant_surface": summarize_quant_surface(tensors),
        "unknown_motifs": bucketed.get("unknown", [])[:64],
        "architecture_hints": extract_architecture_hints(metadata, tensors),
    }
    return report


def print_human(report):
    artifact = report["artifact"]
    print(f"path: {artifact['path']}")
    print(f"target_model: {artifact['target_model']}")
    print(f"gguf_version: {artifact['gguf_version']}")
    print(f"file_size_bytes: {artifact['file_size_bytes']}")
    print()

    hints = report["architecture_hints"]
    print("architecture_hints:")
    for key in (
        "dense_or_moe",
        "hybrid_attention_ssm",
        "has_attention_evidence",
        "has_moe_evidence",
        "has_ssm_evidence",
        "has_deltanet_evidence",
        "has_vision_evidence",
        "has_mtp_evidence",
    ):
        print(f"  {key}: {hints[key]}")
    print()

    inv = report["tensor_inventory"]
    print("tensor_inventory:")
    print(f"  tensor_count: {inv['tensor_count']}")
    print("  quant_histogram:")
    for key, value in inv["quant_histogram"].items():
        print(f"    {key}: {value}")
    print("  top_level_name_families:")
    for key, value in list(inv["top_level_name_families"].items())[:24]:
        print(f"    {key}: {value}")
    print()

    layer_template = report["layer_template"]
    print("layer_template:")
    print(f"  detected_layer_count: {layer_template['detected_layer_count']}")
    print(f"  detected_layer_tensors: {layer_template['detected_layer_tensors']}")
    print("  repeated_suffixes:")
    for key, value in list(layer_template["repeated_suffixes"].items())[:40]:
        print(f"    {key}: {value}")
    print()

    print("metadata_subset:")
    for key, value in report["metadata"].items():
        print(f"  {key}: {json.dumps(value, ensure_ascii=True)}")
    print()

    for section in ("attention_evidence", "moe_evidence", "ssm_evidence", "deltanet_evidence", "state_cache_evidence"):
        print(f"{section}:")
        rows = report[section]
        if not rows:
            print("  []")
        else:
            for item in rows[:20]:
                print(f"  - {item['name']} dims={item['dims']} type={item['type']}")
        print()

    print("extra_heads:")
    for head_name, rows in report["extra_heads"].items():
        print(f"  {head_name}:")
        if not rows:
            print("    []")
        else:
            for item in rows[:20]:
                print(f"    - {item['name']} dims={item['dims']} type={item['type']}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Inspect a Qwen3.6 GGUF and emit a neutral runtime schema view.")
    ap.add_argument("model", help="Path to a Qwen3.6-27B or Qwen3.6-35B-A3B GGUF")
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args()

    try:
        parsed = parse_gguf(args.model)
        report = build_report(parsed)
    except Exception as exc:
        print(f"qwen36_inspect: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

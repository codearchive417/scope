#!/usr/bin/env python3
"""Offline teacher reasoning generation with Qwen3-8B (frozen).

Reads GSM8K (or a dataset with problem/solution columns), generates teacher_reasoning
with the offline REASON_FIRST_PROMPT, optionally filters by answer consistency, and
saves a HuggingFace dataset to disk for scripts/train.py --dataset_path.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json

import torch
from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scope.prompts import build_offline_teacher_user_message
from scope.opd_rc_utils import verify_opd_rc_answer
from scope.data_utils import retained_teacher_records


def normalize_example(example: dict) -> dict:
    problem = example.get("problem") or example.get("question") or example.get("Question")
    solution = example.get("solution") or example.get("answer") or example.get("Answer")
    if problem is None or solution is None:
        raise KeyError(f"Cannot find problem/solution fields in example keys: {list(example.keys())}")
    return {"problem": problem, "solution": solution}


def load_checkpoint_records(checkpoint_path: Path) -> list[dict]:
    if not checkpoint_path.is_file():
        return []
    records: list[dict] = []
    with checkpoint_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_checkpoint_record(checkpoint_path: Path, record: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_shard_range(total: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards}), got {shard_id}")
    base, rem = divmod(total, num_shards)
    start = shard_id * base + min(shard_id, rem)
    end = start + base + (1 if shard_id < rem else 0)
    return start, end


def shard_checkpoint_path(out_path: Path, shard_id: int) -> Path:
    return out_path / "shards" / f"shard_{shard_id:03d}.jsonl"


def migrate_legacy_checkpoint(out_path: Path, num_shards: int, total: int) -> None:
    legacy = out_path / "checkpoint.jsonl"
    if not legacy.is_file():
        return
    if any((out_path / "shards").glob("shard_*.jsonl")):
        return

    records = load_checkpoint_records(legacy)
    if not records:
        return

    print(f"Migrating legacy checkpoint -> shards/ ({num_shards} shards, total={total})")
    out_path.joinpath("shards").mkdir(parents=True, exist_ok=True)
    shard_buffers: dict[int, list[dict]] = {i: [] for i in range(num_shards)}
    for idx, record in enumerate(records):
        record = dict(record)
        record["idx"] = idx
        for shard_id in range(num_shards):
            start, end = get_shard_range(total, shard_id, num_shards)
            if start <= idx < end:
                shard_buffers[shard_id].append(record)
                break

    for shard_id, rows in shard_buffers.items():
        if not rows:
            continue
        path = shard_checkpoint_path(out_path, shard_id)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  shard {shard_id}: migrated {len(rows)} samples -> {path}")


def merge_shards(out_path: Path, num_shards: int, total: int, meta_extra: dict | None = None) -> None:
    records: list[dict | None] = [None] * total
    for shard_id in range(num_shards):
        path = shard_checkpoint_path(out_path, shard_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard checkpoint: {path}")
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = record.get("idx")
            if idx is None:
                raise ValueError(f"Shard record missing idx in {path}")
            if not 0 <= idx < total:
                raise ValueError(f"Shard idx out of range: {idx} (total={total})")
            records[idx] = record

    missing = [i for i, r in enumerate(records) if r is None]
    if missing:
        raise RuntimeError(f"Merge incomplete: missing {len(missing)} indices (e.g. {missing[:5]})")

    complete_records = [record for record in records if record is not None]
    final_records = retained_teacher_records(complete_records)
    if not final_records:
        raise RuntimeError("No answer-verified teacher traces remain after filtering.")

    out_dataset = Dataset.from_list(final_records)
    out_dataset.save_to_disk(str(out_path))

    merged_ckpt = out_path / "checkpoint.jsonl"
    with merged_ckpt.open("w", encoding="utf-8") as f:
        for record in complete_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "num_samples": len(final_records),
        "num_processed": len(complete_records),
        "num_filtered": len(complete_records) - len(final_records),
        "num_shards": num_shards,
        "merged_from_shards": True,
        **(meta_extra or {}),
    }
    with open(out_path / "generation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(final_records)} samples into {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Offline teacher reasoning generation")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Frozen teacher model (default: Qwen3-8B)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="openai/gsm8k",
        help="HuggingFace dataset name (default: openai/gsm8k)",
    )
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local dataset path (overrides dataset_name)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Where to save dataset with teacher_reasoning column",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--filter_failed_answers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop traces that fail the answer check (default: true; paper setting)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint JSONL if present (default: true)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Flush checkpoint JSONL every N new samples (default: 100)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split work across N parallel workers (default: 1)",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="This worker's shard id in [0, num_shards) (default: 0)",
    )
    parser.add_argument(
        "--merge_shards",
        action="store_true",
        help="Merge shard checkpoints into final dataset and exit",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards})")

    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if args.dataset_path:
        local_path = Path(args.dataset_path)
        if local_path.is_dir() and (local_path / "dataset_info.json").is_file():
            split = load_from_disk(str(local_path))
        else:
            raw = load_dataset(args.dataset_path)
            split = raw[args.dataset_split] if args.dataset_split in raw else raw[list(raw.keys())[0]]
    else:
        raw = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
        split = raw

    if args.max_samples is not None:
        split = split.select(range(min(args.max_samples, len(split))))

    total = len(split)

    if args.merge_shards:
        merge_shards(
            out_path,
            args.num_shards,
            total,
            meta_extra={
                "teacher_model": args.model_name_or_path,
                "filter_failed_answers": args.filter_failed_answers,
            },
        )
        return

    migrate_legacy_checkpoint(out_path, args.num_shards, total)

    if args.num_shards == 1:
        checkpoint_path = out_path / "checkpoint.jsonl"
    else:
        checkpoint_path = shard_checkpoint_path(out_path, args.shard_id)

    existing_records: list[dict] = []
    if args.resume:
        existing_records = load_checkpoint_records(checkpoint_path)
        if existing_records:
            print(f"Resuming from {checkpoint_path} ({len(existing_records)} samples done)")
            if args.filter_failed_answers and any(
                "answer_verified" not in record for record in existing_records
            ):
                raise RuntimeError(
                    "This checkpoint predates strict answer filtering. Use a fresh output path "
                    "so failed traces cannot be mistaken for verified traces."
                )

    shard_start, shard_end = get_shard_range(total, args.shard_id, args.num_shards)
    done_indices = {r["idx"] for r in existing_records if "idx" in r}
    if args.num_shards == 1 and existing_records and not done_indices:
        done_indices = set(range(len(existing_records)))

    pending_indices = [idx for idx in range(shard_start, shard_end) if idx not in done_indices]
    if args.num_shards > 1:
        print(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"indices [{shard_start}, {shard_end}) -> {len(pending_indices)} pending"
        )

    if not pending_indices and existing_records:
        print(
            f"Shard {args.shard_id}/{args.num_shards} complete "
            f"({len(existing_records)} samples in {checkpoint_path})"
        )
        if args.num_shards == 1:
            records = existing_records[:total]
            final_records = retained_teacher_records(records)
            if not final_records:
                raise RuntimeError("No answer-verified teacher traces remain after filtering.")
            out_dataset = Dataset.from_list(final_records)
            out_dataset.save_to_disk(str(out_path))
        return

    if args.num_shards == 1 and len(existing_records) >= total:
        print(f"All {total} samples already in checkpoint, writing final dataset...")
        records = existing_records[:total]
        final_records = retained_teacher_records(records)
        if not final_records:
            raise RuntimeError("No answer-verified teacher traces remain after filtering.")
        out_dataset = Dataset.from_list(final_records)
        out_dataset.save_to_disk(str(out_path))
        meta = {
            "teacher_model": args.model_name_or_path,
            "num_samples": len(final_records),
            "num_processed": len(records),
            "filter_failed_answers": args.filter_failed_answers,
            "num_filtered": len(records) - len(final_records),
            "resumed_from_checkpoint": len(existing_records),
        }
        with open(out_path / "generation_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(final_records)} verified samples to {out_path}")
        return

    print(f"Loading teacher model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    records = list(existing_records)
    failed_filter = 0
    batch_size = max(1, args.batch_size)

    desc = "Generating teacher reasoning"
    if args.num_shards > 1:
        desc = f"Shard {args.shard_id}/{args.num_shards}"

    pending_batches = [
        pending_indices[i : i + batch_size]
        for i in range(0, len(pending_indices), batch_size)
    ]
    pbar = tqdm(
        total=shard_end - shard_start,
        desc=desc,
        initial=len(done_indices),
    )
    for batch_indices in pending_batches:
        batch_rows = [normalize_example(split[idx]) for idx in batch_indices]
        prompts = []
        for row in batch_rows:
            user_message = build_offline_teacher_user_message(row["problem"], row["solution"])
            messages = [{"role": "user", "content": user_message}]
            prompts.append(
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, generation_config=gen_config)

        input_width = inputs["input_ids"].shape[1]
        for local_i, idx in enumerate(batch_indices):
            row = batch_rows[local_i]
            problem, solution = row["problem"], row["solution"]
            new_tokens = output_ids[local_i, input_width:]
            teacher_reasoning = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            answer_verified = verify_opd_rc_answer(teacher_reasoning, solution)
            filtered = args.filter_failed_answers and not answer_verified
            if filtered:
                failed_filter += 1

            record = {
                "idx": idx,
                "problem": problem,
                "solution": solution,
                "answer_verified": answer_verified,
                "filtered": filtered,
            }
            if not filtered:
                record["teacher_reasoning"] = teacher_reasoning
            records.append(record)
            append_checkpoint_record(checkpoint_path, record)
        pbar.update(len(batch_indices))
        pbar.set_postfix(batch=batch_size)

    if args.num_shards > 1:
        print(
            f"Shard {args.shard_id}/{args.num_shards} finished "
            f"({len(records)} samples in {checkpoint_path})"
        )
        return

    final_records = retained_teacher_records(records)
    if not final_records:
        raise RuntimeError("No answer-verified teacher traces remain after filtering.")
    out_dataset = Dataset.from_list(final_records)
    out_dataset.save_to_disk(str(out_path))

    meta = {
        "teacher_model": args.model_name_or_path,
        "num_samples": len(final_records),
        "num_processed": len(records),
        "filter_failed_answers": args.filter_failed_answers,
        "num_filtered": len(records) - len(final_records),
        "checkpoint_jsonl": str(checkpoint_path),
        "resumed_from_checkpoint": len(existing_records),
    }
    with open(out_path / "generation_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(final_records)} verified samples to {out_path}")
    if args.filter_failed_answers:
        print(f"Dropped {failed_filter} newly generated traces after failed answer check")


if __name__ == "__main__":
    main()

"""KV-cache evaluation for Qwen2.5-1.5B-Instruct.

Generation stays on the SDPA model. Question-aware scores are a separate FP32
matmul against the rotary-positioned cached keys; the model is not switched to
eager attention. Policies never receive the gold code or the evidence character
spans. Those spans are applied only when measuring how many evidence tokens
were kept.

The case budget is ``M_b`` at the pre-question context length. The question
suffix and newly generated tokens are an FP16 decode component, reported
separately and included in live cache bytes.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from shared.paths import course_data_root
from shared.timing import cuda_sync, time_call
from task2.src.kv_budget import kv_dims_from_config, memory_budget, memory_fp16
from task2.src.kv_cache import RotaryCacheEditor
from task2.src.kv_prompts import policy_fields, question_char_span, tokens_overlapping
from task2.src.kv_quant import pack_channelwise, pack_kv, pack_tokenwise, unpack_tensor
from task2.src.kv_select import select_eviction, select_hybrid, suffix_attention_scores
from task2.src.mqar_data import read_jsonl_gz

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 8
METHODS = (
    "fp16",
    "sink_recent",
    "question",
    "int4_token",
    "int4_channel",
    "hybrid",
)


def retrieval_path() -> Path:
    return course_data_root() / "data" / "kv_retrieval" / "kv_retrieval_cases.jsonl.gz"


def load_cases() -> list[dict]:
    return read_jsonl_gz(retrieval_path())


def extract_five_digits(text: str) -> str | None:
    match = re.search(r"\d{5}", text)
    return match.group(0) if match else None


def kl_from_reference(reference_logits: torch.Tensor, other_logits: torch.Tensor) -> float:
    """KL(softmax(reference) || softmax(other)), mean over the batch."""
    log_p = torch.log_softmax(reference_logits.float(), dim=-1)
    log_q = torch.log_softmax(other_logits.float(), dim=-1)
    return float((log_p.exp() * (log_p - log_q)).sum(dim=-1).mean())


def local_files_only_for(device: torch.device) -> bool:
    """Download only on CUDA. A CPU run must not pull the checkpoint onto a small disk."""
    if os.environ.get("PA4_FORBID_DOWNLOAD") == "1":
        return True
    return device.type != "cuda"


def load_qwen(device: torch.device, local_files_only: bool | None = None):
    """Load the instruct model in FP16. Refuses to download when ``local_files_only`` is set."""
    if local_files_only is None:
        local_files_only = local_files_only_for(device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    n_layers, h_kv, d_h = kv_dims_from_config(model.config)
    if model.config.num_hidden_layers != n_layers or model.config.num_key_value_heads != h_kv:
        raise AssertionError("loaded config does not match the KV dimensions used for the budget")
    return model, tokenizer, (n_layers, h_kv, d_h)


def render_prompt(tokenizer, user_text: str) -> dict:
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    if "offset_mapping" not in encoded:
        raise RuntimeError("tokenizer did not return offset mapping; question spans would be ambiguous")
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    q_start, _q_end = question_char_span(prompt, user_text)
    suffix_tokens = tokens_overlapping(offsets, q_start, len(prompt))
    if not suffix_tokens:
        raise RuntimeError("question suffix produced no tokens")
    context_len = suffix_tokens[0]
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "offsets": offsets,
        "context_len": context_len,
        "prompt_tokens": int(input_ids.shape[1]),
        "question_tokens": int(input_ids.shape[1] - context_len),
    }


def evidence_token_indexes(prompt: str, user_text: str, offsets, start: int, end: int) -> list[int]:
    base = prompt.rfind(user_text)
    if base < 0:
        raise RuntimeError("user text was not found inside the rendered prompt")
    return tokens_overlapping(offsets, base + int(start), base + int(end))


def _stack_context(cache, context_len: int, which: str) -> torch.Tensor:
    pieces = []
    for layer in cache.layers:
        tensor = layer.keys if which == "key" else layer.values
        pieces.append(tensor[0, :, :context_len, :].detach().float().cpu())
    return torch.stack(pieces, dim=0)


def _write_context(cache, restored: torch.Tensor, context_len: int, which: str) -> None:
    for layer_idx, layer in enumerate(cache.layers):
        tensor = layer.keys if which == "key" else layer.values
        tensor[:, :, :context_len, :] = restored[layer_idx].to(device=tensor.device, dtype=tensor.dtype)


def quantize_context(cache, context_len: int, bits: int, key_layout: str, value_layout: str) -> dict:
    keys = _stack_context(cache, context_len, "key")
    values = _stack_context(cache, context_len, "value")
    packed_k, packed_v = pack_kv(keys, values, bits, key_layout, value_layout)
    restored_k = unpack_tensor(packed_k)
    restored_v = unpack_tensor(packed_v)
    report = {
        "k_mean_abs": float((restored_k - keys).abs().mean()),
        "v_mean_abs": float((restored_v - values).abs().mean()),
        "serialized_bytes": packed_k.serialized_bytes() + packed_v.serialized_bytes(),
        "padding_bytes": packed_k.padding_bytes() + packed_v.padding_bytes(),
        "dequantized_bytes": int(restored_k.numel() + restored_v.numel()) * 2,
    }
    _write_context(cache, restored_k, context_len, "key")
    _write_context(cache, restored_v, context_len, "value")
    return report


def quantize_positions(cache, positions: list[int], bits: int, key_layout: str, value_layout: str) -> dict:
    if not positions:
        return {"k_mean_abs": 0.0, "v_mean_abs": 0.0, "serialized_bytes": 0, "padding_bytes": 0, "dequantized_bytes": 0}
    index = torch.tensor(positions, dtype=torch.long)
    keys = torch.stack(
        [layer.keys[0].index_select(1, index).detach().float().cpu() for layer in cache.layers], dim=0
    )
    values = torch.stack(
        [layer.values[0].index_select(1, index).detach().float().cpu() for layer in cache.layers], dim=0
    )
    packed_k, packed_v = pack_kv(keys, values, bits, key_layout, value_layout)
    restored_k = unpack_tensor(packed_k)
    restored_v = unpack_tensor(packed_v)
    for layer_idx, layer in enumerate(cache.layers):
        layer.keys[0].index_copy_(1, index.to(layer.keys.device), restored_k[layer_idx].to(layer.keys.dtype))
        layer.values[0].index_copy_(1, index.to(layer.values.device), restored_v[layer_idx].to(layer.values.dtype))
    return {
        "k_mean_abs": float((restored_k - keys).abs().mean()),
        "v_mean_abs": float((restored_v - values).abs().mean()),
        "serialized_bytes": packed_k.serialized_bytes() + packed_v.serialized_bytes(),
        "padding_bytes": 0,
        "dequantized_bytes": int((restored_k.numel() + restored_v.numel()) * 2),
    }


def question_scores(model, hidden_states, cache, question_start: int) -> torch.Tensor:
    """FP32 attention from the question suffix onto cached, already-rotated keys."""
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    length = hidden_states[0].shape[1]
    position_ids = torch.arange(length, device=hidden_states[0].device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_states[0], position_ids)
    rows = []
    absolute = position_ids[0]
    for layer_idx, layer in enumerate(model.model.layers):
        normed = layer.input_layernorm(hidden_states[layer_idx])
        batch, seq, _ = normed.shape
        width = layer.self_attn.head_dim
        query = layer.self_attn.q_proj(normed).view(batch, seq, -1, width).transpose(1, 2)
        query, _key = apply_rotary_pos_emb(query, query, cos, sin)
        cached_keys = cache.layers[layer_idx].keys
        rows.append(
            suffix_attention_scores(
                query[0],
                cached_keys[0],
                absolute,
                absolute,
                question_start,
            )
        )
    return torch.stack(rows, dim=0)


def gpu_kv_bytes(cache) -> int:
    total = 0
    for layer in cache.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return int(total)


def _forward_token(model, editor: RotaryCacheEditor, token: torch.Tensor) -> torch.Tensor:
    position_ids, cache_position = editor.decode_inputs(token.device)
    with torch.inference_mode():
        output = model(
            input_ids=token,
            past_key_values=editor.cache,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
    editor.note_appended()
    return output.logits[:, -1, :]


def finalize_prompt_logits(model, editor: RotaryCacheEditor, input_ids: torch.Tensor) -> torch.Tensor:
    """Recompute the next-token distribution so it sees the edited cache."""
    last = input_ids[:, -1:]
    editor.drop_last()
    return _forward_token(model, editor, last)


def greedy_continue(model, tokenizer, editor: RotaryCacheEditor, logits: torch.Tensor) -> str:
    generated = []
    eos = tokenizer.eos_token_id
    current = logits
    for _ in range(MAX_NEW_TOKENS):
        nxt = int(current.argmax(dim=-1).item())
        generated.append(nxt)
        if eos is not None and nxt == eos:
            break
        token = torch.tensor([[nxt]], device=current.device)
        current = _forward_token(model, editor, token)
    return tokenizer.decode(generated, skip_special_tokens=True)


def teacher_forced_nll(model, editor: RotaryCacheEditor, prompt_logits: torch.Tensor, answer_ids: list[int]) -> tuple[float, float]:
    """Mean NLL of the gold answer tokens and the mean KL is computed by the caller per step.

    Returns ``(mean_nll, sum_nll)``.
    """
    if not answer_ids:
        return float("nan"), 0.0
    total = 0.0
    current = prompt_logits
    for answer in answer_ids:
        log_probs = torch.log_softmax(current.float(), dim=-1)
        total += float(-log_probs[0, answer])
        token = torch.tensor([[answer]], device=current.device)
        current = _forward_token(model, editor, token)
    return total / len(answer_ids), total


def teacher_forced_kl(
    model,
    reference_editor: RotaryCacheEditor,
    other_editor: RotaryCacheEditor,
    reference_logits: torch.Tensor,
    other_logits: torch.Tensor,
    answer_ids: list[int],
) -> float:
    if not answer_ids:
        return float("nan")
    total = 0.0
    ref_current = reference_logits
    oth_current = other_logits
    for index, answer in enumerate(answer_ids):
        total += kl_from_reference(ref_current, oth_current)
        if index + 1 == len(answer_ids):
            break
        token = torch.tensor([[answer]], device=ref_current.device)
        ref_current = _forward_token(model, reference_editor, token)
        oth_current = _forward_token(model, other_editor, token)
    return total / len(answer_ids)


def evidence_kept_fraction(index_sets: list[list[int]] | None, evidence_tokens: list[int]) -> float | None:
    if index_sets is None or not evidence_tokens:
        return None
    evidence = set(evidence_tokens)
    fractions = []
    for kept in index_sets:
        kept_set = set(kept)
        fractions.append(len(evidence & kept_set) / len(evidence))
    return float(sum(fractions) / len(fractions))


def apply_method(cache, method: str, scores: torch.Tensor | None, context_len: int, prompt_len: int, dims) -> dict:
    """Edit ``cache`` in place. Returns the byte report and the index sets used for evidence."""
    n_layers, h_kv, d_h = dims
    question = list(range(context_len, prompt_len))
    if method == "fp16":
        payload = memory_fp16(context_len, n_layers, h_kv, d_h)
        return {
            "serialized_bytes": payload,
            "context_bytes": payload,
            "k_mean_abs": 0.0,
            "v_mean_abs": 0.0,
            "fp16_index_sets": [list(range(prompt_len)) for _ in range(n_layers)],
            "padding_bytes": 0,
        }
    if method in ("sink_recent", "question"):
        mode = "sink_recent" if method == "sink_recent" else "question"
        context_scores = None if scores is None else scores[:, :context_len]
        chosen, n_keep, total = select_eviction(context_len, n_layers, h_kv, d_h, mode, context_scores)
        full = [sorted(set(layer) | set(question)) for layer in chosen]
        editor_indices = full
        return {
            "serialized_bytes": total,
            "context_bytes": total,
            "k_mean_abs": None,
            "v_mean_abs": None,
            "fp16_index_sets": full,
            "retain_indices": editor_indices,
            "padding_bytes": 0,
            "n_keep_context": n_keep,
        }
    if method == "int4_token":
        report = quantize_context(cache, context_len, 4, "token", "token")
    elif method == "int4_channel":
        report = quantize_context(cache, context_len, 4, "channel", "token")
    elif method == "hybrid":
        if scores is None:
            raise ValueError("hybrid retention needs question scores")
        plan = select_hybrid(scores[:, :context_len], n_layers, h_kv, d_h)
        # Layers share a count. Quantize each layer's own INT2 positions.
        # Byte accounting uses the shared plan; reconstruction uses the real groups.
        report = {"k_mean_abs": 0.0, "v_mean_abs": 0.0, "serialized_bytes": 0, "padding_bytes": 0}
        abs_errors_k = []
        abs_errors_v = []
        serialized = 0
        for layer_idx, positions in enumerate(plan["int2_indices"]):
            layer = cache.layers[layer_idx]
            if not positions:
                continue
            index = torch.tensor(positions, dtype=torch.long, device=layer.keys.device)
            key = layer.keys[0].index_select(1, index).detach().float().cpu().unsqueeze(0)
            value = layer.values[0].index_select(1, index).detach().float().cpu().unsqueeze(0)
            packed_k = pack_channelwise(key, 2)
            packed_v = pack_tokenwise(value, 2)
            restored_k = unpack_tensor(packed_k)[0]
            restored_v = unpack_tensor(packed_v)[0]
            abs_errors_k.append(float((restored_k - key[0]).abs().mean()))
            abs_errors_v.append(float((restored_v - value[0]).abs().mean()))
            serialized += packed_k.serialized_bytes() + packed_v.serialized_bytes()
            layer.keys[0].index_copy_(1, index, restored_k.to(device=layer.keys.device, dtype=layer.keys.dtype))
            layer.values[0].index_copy_(1, index, restored_v.to(device=layer.values.device, dtype=layer.values.dtype))
        fp16_sets = [sorted(set(kept) | set(question)) for kept in plan["fp16_indices"]]
        report = {
            "k_mean_abs": float(sum(abs_errors_k) / len(abs_errors_k)) if abs_errors_k else 0.0,
            "v_mean_abs": float(sum(abs_errors_v) / len(abs_errors_v)) if abs_errors_v else 0.0,
            "serialized_bytes": plan["bytes"],
            "actual_group_bytes": serialized,
            "context_bytes": plan["bytes"],
            "fp16_index_sets": fp16_sets,
            "padding_bytes": 0,
            "budget": plan["budget"],
        }
        if report["serialized_bytes"] > memory_budget(context_len, n_layers, h_kv, d_h, 4):
            raise AssertionError("hybrid exceeds the case budget")
        return report
    else:
        raise ValueError(f"unknown cache method {method}")
    report["context_bytes"] = report["serialized_bytes"]
    report["fp16_index_sets"] = None
    budget = memory_budget(context_len, n_layers, h_kv, d_h, 4)
    if method.startswith("int4") and report["serialized_bytes"] != budget:
        raise AssertionError(
            f"{method} packed {report['serialized_bytes']} bytes, budget is {budget}"
        )
    return report


def prefill(model, input_ids: torch.Tensor):
    return model(input_ids, use_cache=True, output_hidden_states=True)


def evaluate_case(model, tokenizer, row: dict, dims, device: torch.device) -> list[dict]:
    """One prompt under the FP16 reference and the five compressed methods."""
    visible = policy_fields(row)
    rendered = render_prompt(tokenizer, visible["user_text"])
    input_ids = rendered["input_ids"].to(device)
    cuda_sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cuda_sync()
    with torch.inference_mode():
        first = prefill(model, input_ids)
    cuda_sync()
    prefill_seconds = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
    score_started = time.perf_counter()
    scores = question_scores(model, first.hidden_states, first.past_key_values, rendered["context_len"])
    cuda_sync()
    score_seconds = time.perf_counter() - score_started
    reference_logits = first.logits[:, -1, :].detach()
    base_cache = first.past_key_values
    del first
    answer_ids = tokenizer(str(row["code"]), add_special_tokens=False)["input_ids"]
    evidence = evidence_token_indexes(
        rendered["prompt"],
        visible["user_text"],
        rendered["offsets"],
        row["evidence_char_start"],
        row["evidence_char_end"],
    )
    records = []
    reference_exact = None
    for method in METHODS:
        editor = RotaryCacheEditor(base_cache, input_ids.shape[1]).clone()
        edited = apply_method(
            editor.cache,
            method,
            scores.cpu(),
            rendered["context_len"],
            rendered["prompt_tokens"],
            dims,
        )
        if "retain_indices" in edited:
            editor.retain(edited["retain_indices"])
        decode_started = time.perf_counter()
        prompt_logits = finalize_prompt_logits(model, editor, input_ids)
        if method == "fp16":
            fp16_logits = prompt_logits.detach()
            fp16_editor = editor.clone()
            post_kl = 0.0
            forced_kl = 0.0
        else:
            post_kl = kl_from_reference(fp16_logits, prompt_logits)
            forced_kl = teacher_forced_kl(
                model,
                fp16_editor.clone(),
                editor.clone(),
                fp16_logits,
                prompt_logits.detach(),
                answer_ids,
            )
        greedy_editor = editor.clone()
        nll_editor = editor.clone()
        text = greedy_continue(model, tokenizer, greedy_editor, prompt_logits)
        mean_nll, _sum_nll = teacher_forced_nll(model, nll_editor, prompt_logits, answer_ids)
        cuda_sync()
        decode_seconds = time.perf_counter() - decode_started
        extracted = extract_five_digits(text)
        exact = int(extracted == str(row["code"]))
        if method == "fp16":
            reference_exact = exact
        question_bytes = memory_fp16(rendered["question_tokens"], *dims)
        generated_tokens = min(MAX_NEW_TOKENS, 8)
        generated_bytes = memory_fp16(generated_tokens, *dims)
        records.append(
            {
                "id": row["id"],
                "method": method,
                "target_tokens": row["target_tokens"],
                "position": row["position"],
                "distractors": row["distractors"],
                "prompt_tokens": rendered["prompt_tokens"],
                "context_tokens": rendered["context_len"],
                "question_tokens": rendered["question_tokens"],
                "exact": exact,
                "extracted": extracted,
                "reference_exact": reference_exact if method == "fp16" else None,
                "nll": mean_nll,
                "post_prefill_kl": post_kl,
                "teacher_forced_kl": forced_kl,
                "evidence_fraction": evidence_kept_fraction(edited.get("fp16_index_sets"), evidence),
                "serialized_bytes": edited["serialized_bytes"],
                "question_fp16_bytes": question_bytes,
                "generated_fp16_bytes": generated_bytes,
                "live_bytes": edited["serialized_bytes"] + question_bytes + generated_bytes,
                "gpu_kv_bytes": gpu_kv_bytes(editor.cache),
                "k_mean_abs": edited.get("k_mean_abs"),
                "v_mean_abs": edited.get("v_mean_abs"),
                "score_seconds": score_seconds if method in ("question", "hybrid") else 0.0,
                "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds,
                "peak_allocated_bytes": peak,
                "finite_logits": bool(torch.isfinite(prompt_logits).all()),
                "generation": text,
            }
        )
    for record in records:
        record["reference_exact"] = reference_exact
    if not all(record["finite_logits"] for record in records):
        raise RuntimeError(f"non-finite logits on {row['id']}")
    return records


def split_matches_full(model, input_ids: torch.Tensor) -> dict:
    """C1: a full forward and a prefill-plus-last-token decode agree, and logits are finite."""
    full = model(input_ids, use_cache=False).logits
    prefix = model(input_ids[:, :-1], use_cache=True)
    editor = RotaryCacheEditor(prefix.past_key_values, input_ids.shape[1] - 1)
    step_logits = _forward_token(model, editor, input_ids[:, -1:])
    # Keeping every token must preserve the original rotary ids 0..L-2 and then append L-1.
    kept = editor.position_ids[0].tolist()
    delta = (full[:, -1, :].float() - step_logits.float()).abs().max()
    return {
        "max_abs": float(delta),
        "finite": bool(torch.isfinite(full).all() and torch.isfinite(step_logits).all()),
        "position_ids": kept,
    }


def benchmark_pack_unpack(n_layers: int, h_kv: int, d_h: int, n_q: int, device: torch.device) -> list[dict]:
    """Ten warmed-up repetitions at batches 1 and 8 and context lengths 2048 and 4096."""
    rows = []
    for length in (2048, 4096):
        for batch in (1, 8):
            keys = torch.randn(batch, n_layers, h_kv, length, d_h, device=device)
            values = torch.randn_like(keys)
            query = torch.randn(batch, n_q, 1, d_h, device=device)

            def _pack(keys=keys, values=values):
                return pack_kv(keys, values, 4, "channel", "token")

            packed = _pack()
            serialized = sum(item.serialized_bytes() for pair in [packed] for item in pair)

            def _unpack(packed=packed, device=device):
                return unpack_tensor(packed[0]).to(device), unpack_tensor(packed[1]).to(device)

            def _fp16_decode(query=query, keys=keys, values=values):
                layer_q = query
                layer_k = keys[:, 0]
                layer_v = values[:, 0]
                return F.scaled_dot_product_attention(layer_q, layer_k, layer_v, is_causal=False, enable_gqa=True)

            def _quant_decode(query=query, packed=packed, device=device):
                layer_k = unpack_tensor(packed[0]).to(device)[:, 0]
                layer_v = unpack_tensor(packed[1]).to(device)[:, 0]
                return F.scaled_dot_product_attention(query, layer_k, layer_v, is_causal=False, enable_gqa=True)

            rows.append(
                {
                    "batch": batch,
                    "length": length,
                    "serialized_bytes": serialized,
                    "dequantized_bytes": int(keys.numel() + values.numel()) * keys.element_size(),
                    "pack_seconds": time_call(_pack, warmup=10, repeats=10),
                    "unpack_seconds": time_call(_unpack, warmup=10, repeats=10),
                    "fp16_decode_seconds": time_call(_fp16_decode, warmup=10, repeats=10),
                    "unpack_decode_seconds": time_call(_quant_decode, warmup=10, repeats=10),
                }
            )
    return rows


def run_kv_suite(device: torch.device | None = None, limit: int | None = None) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, dims = load_qwen(device)
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]
    records = []
    for row in cases:
        records.extend(evaluate_case(model, tokenizer, row, dims, device))
    return records


def write_records(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")

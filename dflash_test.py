import sys, os, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
from huggingface_hub import snapshot_download
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from common import format_prompt, get_stop_conditions

PROMPT_FORMAT = "gemma4"
SYSTEM = "You are a helpful assistant."
INSTRUCTION = "Explain why the sky is blue in three short paragraphs."
MAX_NEW = 200
CACHE_SIZE = 8192
DEVICE = "cuda:0"

target_dir = snapshot_download("turboderp/gemma-4-31b-it-exl3", revision="6.00bpw")
draft_dir = snapshot_download("turboderp/gemma4-31b-it-DFlash-exl3", revision="6.00bpw")
print(f"target: {target_dir}")
print(f"draft : {draft_dir}")

# --- target ---
config = Config.from_directory(target_dir)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)

# --- draft (DFlash) ---
draft_config = Config.from_directory(draft_dir)
draft_model = Model.from_config(draft_config)
draft_default = draft_model.caps.get("default_draft_size", 0)
print(f"draft caps: dflash_draft={draft_model.caps.get('dflash_draft')} "
      f"default_draft_size={draft_default}")

draft_cache = Cache(draft_model, max_num_tokens=CACHE_SIZE)
draft_model.load(progressbar=True, device=DEVICE)

cache = Cache(model, max_num_tokens=CACHE_SIZE, max_batch_size=1,
              max_history=draft_default)
model.load(progressbar=True, device=DEVICE)


def run(use_draft, label):
    gen = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        draft_model=draft_model if use_draft else None,
        draft_cache=draft_cache if use_draft else None,
    )
    formatted = format_prompt(PROMPT_FORMAT, SYSTEM, INSTRUCTION)
    job = Job(
        input_ids=tokenizer.encode(formatted, add_bos=True),
        max_new_tokens=MAX_NEW,
        stop_conditions=get_stop_conditions(PROMPT_FORMAT, tokenizer),
        sampler=ArgmaxSampler(),
    )
    gen.enqueue(job)
    text, ntok, acc, rej = "", 0, 0, 0
    t0 = time.time()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            text += r.get("text", "")
            if r.get("eos"):
                ntok = r.get("new_tokens", ntok)
                acc = r.get("accepted_draft_tokens", 0)
                rej = r.get("rejected_draft_tokens", 0)
    dt = time.time() - t0
    print(f"\n=== {label} ===")
    print(f"tokens={ntok} time={dt:.2f}s decode={ntok/dt:.2f} tok/s")
    if use_draft:
        total = acc + rej
        ar = (acc / total * 100) if total else 0.0
        print(f"accepted={acc} rejected={rej} acceptance={ar:.1f}%")
    print("OUTPUT:\n" + text)
    return text


base = run(False, "BASELINE (no draft)")
draft = run(True, "DFLASH draft")
print("\n=== MATCH ===")
print("identical:" , base == draft)
if base != draft:
    n = min(len(base), len(draft))
    i = next((j for j in range(n) if base[j] != draft[j]), n)
    print(f"first diff at char {i}")
    print("base :", repr(base[max(0,i-40):i+40]))
    print("draft:", repr(draft[max(0,i-40):i+40]))

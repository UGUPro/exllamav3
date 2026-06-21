import sys, os, time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"))
from huggingface_hub import snapshot_download
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from common import format_prompt, get_stop_conditions

PROMPT_FORMAT = "gemma4"
SYSTEM = "You are a helpful assistant."
CACHE_SIZE = 8192
DEVICE = "cuda:0"

PROMPTS = {
    "hard (reasoning)": "Explain why the sky is blue in three short paragraphs.",
    "easy (count)":     "Count from 1 to 60, one number per line, like '1\\n2\\n3'. Just the numbers.",
    "easy (repeat)":    "Repeat the following sentence exactly 20 times: The quick brown fox jumps over the lazy dog.",
}

target_dir = snapshot_download("turboderp/gemma-4-31b-it-exl3", revision="6.00bpw")
draft_dir = snapshot_download("turboderp/gemma4-31b-it-DFlash-exl3", revision="6.00bpw")

config = Config.from_directory(target_dir)
model = Model.from_config(config)
tokenizer = Tokenizer.from_config(config)
draft_config = Config.from_directory(draft_dir)
draft_model = Model.from_config(draft_config)
draft_default = draft_model.caps.get("default_draft_size", 0)
draft_cache = Cache(draft_model, max_num_tokens=CACHE_SIZE)
draft_model.load(progressbar=True, device=DEVICE)
cache = Cache(model, max_num_tokens=CACHE_SIZE, max_batch_size=1, max_history=draft_default)
model.load(progressbar=True, device=DEVICE)

gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                draft_model=draft_model, draft_cache=draft_cache)
print(f"num_draft_tokens = {gen.num_draft_tokens}")

for label, instr in PROMPTS.items():
    formatted = format_prompt(PROMPT_FORMAT, SYSTEM, instr)
    job = Job(input_ids=tokenizer.encode(formatted, add_bos=True),
              max_new_tokens=200,
              stop_conditions=get_stop_conditions(PROMPT_FORMAT, tokenizer),
              sampler=ArgmaxSampler())
    gen.enqueue(job)
    ntok = acc = rej = 0
    t0 = time.time()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("eos"):
                ntok = r.get("new_tokens", 0)
                acc = r.get("accepted_draft_tokens", 0)
                rej = r.get("rejected_draft_tokens", 0)
    dt = time.time() - t0
    tot = acc + rej
    ar = (acc / tot * 100) if tot else 0.0
    print(f"[{label:18s}] tok={ntok:3d} {ntok/dt:5.2f} tok/s | "
          f"accept={ar:5.1f}% (acc={acc} rej={rej})")

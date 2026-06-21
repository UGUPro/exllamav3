import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job

# exl3 model dir (HF cache snapshot or a local exl3 folder); override with argv[1]
md = sys.argv[1] if len(sys.argv) > 1 else "/home/tao/.cache/huggingface/hub/models--dr-housemd--G4-MeroMero-31B-uncensored-heretic-exl3-6.00bpw/snapshots/7b82450ba79f922f8870540cd226a1f0206c3369"
import flash_attn
from exllamav3.modules.attention_fn.flash_attn_2 import has_flash_attn
print(f"[test] flash_attn {flash_attn.__version__}; dispatch has_flash_attn={has_flash_attn}", flush=True)

cfg = Config.from_directory(md)
model = Model.from_config(cfg)
# Bigger cache + force enough pages that the second sequence reuses/relocates pages.
cache = Cache(model, max_num_tokens=32768)
model.load()
tok = Tokenizer.from_config(cfg)
gen = Generator(model=model, cache=cache, tokenizer=tok)

def run(prompt, n=20):
    ids = tok.encode(prompt, add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=n)
    gen.enqueue(job)
    out = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r["stage"] == "streaming":
                out += r.get("text", "")
    return out

# Reuse the SAME generator/cache across multiple prompts (server-like usage).
for p in ["The capital of France is",
          "Write one sentence about the sea:",
          "Q: What color is a banana? A:",
          "2 + 2 ="]:
    print(f"[{p!r}] -> {run(p)!r}", flush=True)

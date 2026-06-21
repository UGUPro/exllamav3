import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
try:
    import flash_attn
    from exllamav3.modules.attention_fn.flash_attn_2 import has_flash_attn
    print(f"[test] flash_attn {flash_attn.__version__} importable; dispatch has_flash_attn={has_flash_attn}")
except Exception as e:
    print("[test] no flash_attn:", e)
md = sys.argv[1]
cfg = Config.from_directory(md)
model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens=4096)
model.load()
tok = Tokenizer.from_config(cfg)
gen = Generator(model=model, cache=cache, tokenizer=tok)
ids = tok.encode("The capital of France is", add_bos=True)
job = Job(input_ids=ids, max_new_tokens=24)
gen.enqueue(job)
out = ""
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r["stage"] == "streaming":
            out += r.get("text", "")
print("[test] OUTPUT:", repr(out))

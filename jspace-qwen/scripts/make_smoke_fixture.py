"""Build an offline smoke-test fixture: a tiny random Qwen3 + a byte-level BPE
tokenizer trained on the released prompt sets, saved as a local HF model dir,
plus a small corpus JSON. Lets every script run end-to-end without downloads.

python scripts/make_smoke_fixture.py OUT_DIR
"""

import glob
import json
import os
import sys

import torch
import transformers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = __file__.rsplit("/scripts/", 1)[0]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)


def strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from strings(v)


texts = []
for f in glob.glob(os.path.join(ROOT, "data", "*", "*.json")):
    texts += list(strings(json.load(open(f))))
corpus = [" ".join(texts[i:i + 40]) for i in range(0, len(texts), 40)]

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
tok.train_from_iterator(texts * 3, trainers.BpeTrainer(vocab_size=3000, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
fast.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                      "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
fast.save_pretrained(out)

cfg = transformers.AutoConfig.for_model(
    "qwen3", vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
    tie_word_embeddings=True)
torch.manual_seed(0)
model = transformers.AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
model.save_pretrained(out)
with open(os.path.join(out, "corpus.json"), "w") as f:
    json.dump([c for c in corpus if len(c) > 600] * 4, f)
print("fixture at", out, "vocab", len(fast), "corpus records", len(corpus))

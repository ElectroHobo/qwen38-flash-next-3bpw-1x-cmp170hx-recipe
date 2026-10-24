# Model and checkpoint licences

This recipe downloads weights it does not own and combines two checkpoints into the
one it serves. Read the terms before serving them to anyone else.

| Component | Where it comes from | Licence |
|---|---|---|
| Qwen3.8-Flash-Next (the model) | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Qwen Community License 1.0 (both checkpoints ship it as `LICENSE`) |
| The weights served here, 2- and 3-bit AutoRound with an MTP head | [klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP](https://huggingface.co/klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP), revision `ce0e0b94`, without its BF16 n-gram table shard | the card says `license: other`, Qwen Community License 1.0 |
| The FP8 n-gram (PLE) table | the `model-plefp8-*` shards of [halt95/Qwen3.8-Flash-Next-W4A16-Merlin](https://huggingface.co/halt95/Qwen3.8-Flash-Next-W4A16-Merlin), revision `cec53e9f`; quantised by [RadixArk](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Qwen Community License 1.0; as the RadixArk card states |
| The served checkpoint | built on your disk by `docker/build_checkpoint.py` from the two above, hard links and a merged index; not redistributed by this recipe | the terms of its parts |
| The DFlash 2 drafter (the default) | [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2), revision `a22a1744` | Apache-2.0 |
| The patched engine files in `patches/` | Morrowmake's vllm-cmp170hx, modified, and a new package | Apache-2.0, headers kept, changes noted |
| The engine | [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) | Apache-2.0 |
| The engine image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6…` | Apache-2.0, with the image's `THIRD_PARTY_NOTICES` |
| This recipe's scripts | Morrowmake's Qwen3.8-Flash-Next recipe, modified (by way of our two-card recipe) | MIT |
| Benchmark prompts | MiaAI-Lab's kits, used as data with attribution | see [results.md](results.md), "Notes on the method" |

inco.ai's Qwen drafter is Apache-2.0 — unlike their GLM drafter, which is CC
BY-NC-ND, so check the licence before swapping drafters.

klee100's card credits Qwen and Intel's AutoRound; halt95's credits Qwen, Intel,
RadixArk, DominikBucko, the vLLM community's PLE-offload work and aikitoria; this
recipe passes those credits on. The served checkpoint stays on your machine; if you
redistribute it or weights derived from it, the Qwen Community License's
attribution and naming terms apply to you.

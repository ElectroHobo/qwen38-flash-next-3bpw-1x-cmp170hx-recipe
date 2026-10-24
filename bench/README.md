# Benchmarks

Stdlib-only Python against a running server. Each takes a port and a model id;
run them on an idle server. The raw output of every run behind
[`docs/results.md`](../docs/results.md) is in `results/release-0.1.0/`, the development
runs in its `development/` folder.

| Script | Measures | Takes |
|---|---|---|
| `cmp_recipe_bench.py --port 8000 --model qwen3.8-flash-next --tag mine` | decode at 1 / 2 / 4 / 8 users and cold prefill, the MiaAI-Lab protocol Morrowmake's tables use; writes a markdown table and a JSON of the runs | ~5 min |
| `concurrency_sweep.py --port 8000 --model qwen3.8-flash-next --tag mine` | aggregate decode tok/s, ms per engine step and tokens per request-step from 1 to 128 users (`--users`; this recipe takes 48), 700-token replies at temperature 0, the three task prompts rotating; writes `results/concurrency-<tag>.json` | ~10 min |
| `vllm_conc_steps.py 8000 qwen3.8-flash-next` | engine steps, ms per step and draft acceptance per position at 1 / 2 / 4 users, from the server's own counters | ~3 min |
| `kv_offload_bench.py mine --port 8000 --model qwen3.8-flash-next` | whether a 100k-token context evicted by a 1.8M-token flood comes back from the KV tier (time to first token, hidden codes); compare a run with `KV_OFFLOAD_GB=0` against one with the tier | ~8 min |
| `kv_dtype_check.py mine --port 8000 --model qwen3.8-flash-next` | pool size, 100k-context prefill and decode, five temperature-0 answers to diff between two KV dtypes | ~3 min |
| `prefill_fairness.py mine --port 8000 --model qwen3.8-flash-next` | what a decoding user keeps while a 30k prompt is read: tok/s before / during / after, longest pause | ~1 min |
| `soak.py` | sustained multi-turn load with image requests mixed in (from the MiMo recipe; see its header) | as configured |
| `agent_probes.py edit mine --port 8000 --model qwen3.8-flash-next` | code editing: a 2k-token file in the prompt, rename every function, return the whole file; decode tok/s from `usage` | ~1 min |
| `agent_probes.py followup mine --port 8000 --model qwen3.8-flash-next` | follow-up turns: an ~18k-token conversation's second question; time to first token for each turn | ~1 min |
| `quality_eval.py run --base http://127.0.0.1:8000 --tag mine --suites humaneval gsm8k`, then `score` and `report` | HumanEval (164, the dataset's tests run in a network-less container) and GSM8K (first 200), thinking on; needs Docker on the machine running the harness | ~10 min |

`kv_offload_bench.py` sends its questions through the chat endpoint with
thinking off: this model answers a raw completion prompt with reasoning tokens
only, so the hidden-code check would read empty.

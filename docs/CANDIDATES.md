# Model candidate analysis

Collected 2026-11-07 on host `ub2404srv` (RTX 5000 Ada, 32760 MiB VRAM, 91.9
GiB RAM, 653 GiB free disk, Ollama 0.33.2). Installed-model facts come from
Ollama's `/api/tags` and `/api/show`. Download sizes come from read-only
manifest lookups against `registry.ollama.ai` — no model was downloaded.

Sizes are quoted in the unit the source uses: Ollama's API reports decimal GB,
and manifests sum to decimal GB, while VRAM is MiB/GiB. Resident-size figures
marked *estimate* add 15% to on-disk size for KV cache and runtime buffers.

## The four requested candidates

### 1. `gpt-oss:20b` — installed, ready

| Field | Value | Source |
|---|---|---|
| Installed | yes | `/api/tags` |
| Size | 13.79 GB (12.85 GiB) | `/api/tags` |
| Registry size | 13.79 GB — identical, so the local copy is current | manifest |
| Quantization | MXFP4 | `/api/show` |
| Architecture | `gptoss` | `/api/show` |
| MoE | yes, 4 of 32 experts per token | `expert_count` / `expert_used_count` |
| Parameters | 20.9B total; active count not reported | `/api/show` |
| Context capability | 131072 | `gptoss.context_length` |
| Capabilities | completion, tools, thinking | `/api/show` |
| Vision | no | `/api/show` |
| Estimated resident | 14.8 GiB — comfortably inside 32 GiB | estimate |
| Likely residency | fully on GPU, with headroom for long context | estimate |

A second tag, `gpt-oss-64k:20b`, has the same size and metadata and appears to
be a context-length variant. Benchmark the plain tag first.

### 2. `qwen3:30b` — **not installed**; the tag does exist

| Field | Value | Source |
|---|---|---|
| Installed | no | `/api/tags` |
| Registry tag exists | yes | manifest |
| Download | 18.56 GB, single model layer | manifest |
| `qwen3:30b-a3b` | same manifest size as `qwen3:30b`, i.e. an alias of the same build | manifest |
| Expected architecture | `qwen3moe` (matches the installed derivatives) | inference from local variants, **not verified for this tag** |

What *is* installed from the same base: `qwen3-coder:30b` and
`qwen3-coder-abliterated-tools:30b` (coder variants) and
`huihui_ai/qwen3-abliterated:30b` (abliterated instruct), all 18.56 GB,
`qwen3moe`, 8 of 128 experts, 262144 context. The official instruct tag itself
is absent.

Verdict: worth downloading. 18.56 GB of 653 GB free, and it is the only way to
compare the official instruct build against the coder and abliterated variants
already present.

### 3. `qwen3.5:35b-a3b` → resolves to `qwen3.5:35b`, which **is installed**

| Field | Value | Source |
|---|---|---|
| Installed as | `qwen3.5:35b` | `/api/tags` |
| Local size | 23.87 GB (22.23 GiB) | `/api/tags` |
| Registry manifest for `:35b` and `:35b-a3b` | both total 22.07 GB (21.17 GB model + 0.90 GB vision projector) | manifest |
| Size mismatch | local 23.87 GB vs registry 22.07 GB — **the installed copy is not the same build as the current registry tag** | comparison |
| Quantization | Q4_K_M | `/api/show` |
| Architecture | `qwen35moe` | `/api/show` |
| MoE | yes, 8 of 256 experts per token | `/api/show` |
| Parameters | 36.0B total; active count not reported | `/api/show` |
| Context capability | 262144 | `qwen35moe.context_length` |
| Capabilities | completion, **vision**, tools, thinking | `/api/show` |
| Estimated resident | 25.6 GiB of 32 GiB — fits, but tight once KV cache grows | estimate |
| Likely residency | full GPU residency at modest context; long context will push it to offload | estimate |

Two notes. First, `-a3b` is an alias here: both tags resolve to the same
manifest size, so there is nothing separate to download. Second, the local
copy predates or postdates the current registry build; re-pulling would change
the digest and break comparability with any earlier results. Benchmark what is
installed, and only re-pull deliberately.

Closely related tags already installed: `qwen3.6:35b`, `qwen3.6-64k:35b`,
`qwen3.6-abliterated-64k:35b` (all 23.94 GB, `qwen35moe`, 8/256) and
`ornith-1.5:35b` / `ornith-1.5-64k:35b` (22.62 GB, `qwen35moe`, 8/256).

### 4. `gpt-oss:120b` — optional experiment, not a benchmark target yet

| Field | Value | Source |
|---|---|---|
| Installed | no | `/api/tags` |
| Download | 65.37 GB | manifest |
| Disk impact | 65.37 GB of 653 GB free (10%) | calculation |
| Estimated resident | ~70 GiB against 32 GiB VRAM | estimate |
| Likely residency | roughly 55-60% of weights offloaded to system RAM | estimate |
| RAM feasibility | ~70 GiB resident against 91.9 GiB total / 67.6 GiB currently available — feasible but with little margin, and swap is already 7.9 of 8.0 GiB used | measurement + estimate |

Do not download this yet. It would be the slowest configuration on this host
by a wide margin and the memory margin is thin. It is only worth doing as a
deliberate "how bad is heavy offload here?" experiment, after the fast
candidates have produced baseline numbers to compare against.

## Installed models worth benchmarking alongside the candidates

19 of the 60 installed tags are MoE. The ones that look most competitive for
the workloads in question:

| Model | Size | Arch | Experts | Context | Capabilities | Why |
|---|---|---|---|---|---|---|
| `gpt-oss:20b` | 13.79 GB | gptoss | 4/32 | 131072 | tools, thinking | Smallest strong MoE here; best speed/VRAM headroom |
| `qwen3-coder:30b` | 18.56 GB | qwen3moe | 8/128 | 262144 | tools | Purpose-built coder, MoE, fits easily |
| `qwen3.5:35b` | 23.87 GB | qwen35moe | 8/256 | 262144 | tools, thinking, **vision** | Largest that still fits; the vision-capable candidate |
| `qwen3.6:35b` | 23.94 GB | qwen35moe | 8/256 | 262144 | tools, thinking, vision | Newer sibling of the above; direct generational comparison |
| `glm-4.7-flash:latest` | 19.02 GB | glm4moelite | 4/64 | 202752 | tools, thinking | Different architecture family — guards against a Qwen-only conclusion |
| `qwen3.8:27b` | 17.74 GB | qwen35 (dense) | — | 262144 | tools, thinking, vision | Dense control for the MoE comparison at similar size |
| `qwen3:32b` | 20.20 GB | qwen3 (dense) | — | 40960 | tools, thinking | Second dense control; note the much shorter context |
| `huihui_ai/qwen3-vl-abliterated:30b-a3b-instruct` | 19.60 GB | qwen3vlmoe | 8/128 | 262144 | tools, vision | Vision MoE alternative |

Excluded from benchmarking, with reasons:

- `minimax-m2:cloud` — Ollama reports 0 bytes and no architecture. This is not
  local weights; it is a cloud-hosted tag and cannot be measured as local
  inference.
- `mxbai-embed-large:335m`, `nomic-embed-text:latest` — embedding models; the
  suite does not apply.
- `hf.co/mradermacher/Anubis-70B-v1.2-i1-GGUF:IQ3_XS` — 29.31 GB, estimated
  31.4 GiB resident against 32 GiB VRAM, so it will offload. Interesting only
  as an offload data point, not as a primary candidate.
- `darkest-universe:29b` reports a 1,024,000-token context capability, which
  is worth a sanity check before trusting any long-context result from it.
- Roleplay, uncensored and abliterated variants, and the three
  `whisperjav-*` tags — not relevant to coding, reasoning or tool use, beyond
  the abliterated Qwen builds kept above as same-base comparisons.

## Proposed experiment order

Each stage is a separate approved run. Stop and look at the results before
moving on.

1. **Smoke test, one model, one category.** `gpt-oss:20b`, instruction
   following only, 2 repetitions. Confirms the real adapter, telemetry,
   residency reporting and grading work against a live engine. A few minutes.
2. **Full suite, smallest candidate.** `gpt-oss:20b`, all categories, 3
   repetitions, code execution enabled. Establishes the baseline and the real
   per-attempt cost, which replaces the current cost estimate.
3. **The three fast installed candidates.** `gpt-oss:20b`,
   `qwen3-coder:30b`, `qwen3.5:35b`, full suite. This is the comparison that
   answers the original question.
4. **Architecture and density controls.** Add `glm-4.7-flash:latest` and the
   dense `qwen3.8:27b`, full suite.
5. **Context scaling.** Re-run the long-context category on the top two models
   at `--context 16384`, `32768` and `65536`, watching `gpu_fraction` for the
   point where offload begins.
6. **Download `qwen3:30b`** (18.56 GB) and compare it against the coder and
   abliterated variants of the same base.
7. **Optional offload experiment.** `gpt-oss:120b` (65.37 GB) or the 70B
   IQ3_XS tag, long-context category only, with a raised timeout, purely to
   quantify how badly offload hurts on this host.

## Proposed download plan

Nothing is downloaded without approval. In order of value:

| Order | Model | Download | Disk after | Expected residency | Rationale |
|---|---|---|---|---|---|
| 1 | `qwen3:30b` | 18.56 GB | ~634 GB free | fully on GPU (estimate) | Only missing priority candidate; official instruct build |
| 2 | *(optional)* `gpt-oss:120b` | 65.37 GB | ~569 GB free | heavy CPU offload (estimate) | Deliberate offload experiment only |

Total if both: 83.93 GB, leaving roughly 569 GB free. Neither is required for
stages 1-5, which use only installed models.

## Resource and compatibility notes

- Ollama 0.33.2 is running with `OLLAMA_MAX_LOADED_MODELS=1` and
  `OLLAMA_NUM_PARALLEL=4`. The single-model limit suits serial benchmarking
  well. `OLLAMA_NUM_PARALLEL=4` means the server may split the KV cache budget
  across four slots, which can affect effective context per request;
  Caliper records the `num_ctx` it asked for and the prompt tokens actually
  processed, so this shows up in results rather than being hidden. Changing
  that setting is a service change and has not been done.
- Swap is 7.9 GiB used of 8.0 GiB with 67.6 GiB RAM available. Nothing is
  under memory pressure now, but a heavily offloaded model has very little
  swap to fall back on.
- One pre-existing GPU process (pid 2138, ComfyUI, 430 MiB) has held the
  device for three days. It is reported by preflight and never disturbed; its
  430 MiB counts against available VRAM.
- `qwen3:32b` reports only 40960 tokens of context, far less than the
  262144 of the Qwen3 MoE builds. Do not compare long-context results across
  those two without noting it.
- vLLM 0.11.0 is installed in a separate conda environment
  (Python 3.11.13, torch 2.8.0+cu128) and is
  not running. It cannot load Ollama's GGUF/MXFP4 artifacts directly, so a
  like-for-like comparison needs separately obtained weights. See
  [ROADMAP.md](ROADMAP.md).

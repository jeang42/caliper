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

### 2. `qwen3:30b` — **not installed**; the tag does exist

| Field | Value | Source |
|---|---|---|
| Installed | no | `/api/tags` |
| Registry tag exists | yes | manifest |
| Download | 18.56 GB, single model layer | manifest |
| `qwen3:30b-a3b` | same manifest size as `qwen3:30b`, i.e. an alias of the same build | manifest |
| Expected architecture | `qwen3moe` (matches the installed derivatives) | inference from local variants, **not verified for this tag** |

What *is* installed from the same base is a coder build,
`qwen3-coder:30b` — 18.56 GB, `qwen3moe`, 8 of 128 experts, 262144 context.
The official instruct tag itself is absent.

Verdict: worth downloading. 18.56 GB of 653 GB free, and it is the only way to
compare the official instruct build against the coder build on the same
base.

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

Newer siblings on the same architecture are installed locally too, so a
generational comparison is possible without a download.

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

## The models actually benchmarked

These five are the ones measured; their figures are what
[RESULTS.md](RESULTS.md) reports. Other locally installed tags were considered
and set aside, and that longer inventory is deliberately not published here:
embedding models the suite does not apply to, a cloud-hosted tag that is not
local inference at all, roleplay and uncensored variants irrelevant to coding,
reasoning and tool use, and one 29 GB build whose estimated 31.4 GiB resident
size against 32 GiB of VRAM makes it an offload data point rather than a
candidate.

| Model | Size | Arch | Experts | Context | Capabilities | Why it is in the set |
|---|---|---|---|---|---|---|
| `gpt-oss:20b` | 13.79 GB | gptoss | 4/32 | 131072 | tools, thinking | Smallest strong MoE here; best speed/VRAM headroom |
| `qwen3-coder:30b` | 18.56 GB | qwen3moe | 8/128 | 262144 | tools | Purpose-built coder, MoE, fits easily |
| `qwen3.5:35b` | 23.87 GB | qwen35moe | 8/256 | 262144 | tools, thinking, **vision** | Largest that still fits; the vision-capable candidate |
| `glm-4.7-flash:latest` | 19.02 GB | glm4moelite | 4/64 | 202752 | tools, thinking | Different architecture family — guards against a Qwen-only conclusion |
| `qwen3.8:27b` | 17.74 GB | qwen35 (dense) | — | 262144 | tools, thinking, vision | Dense control for the MoE comparison at similar size |

Selection criteria, so the set can be reproduced rather than taken on trust:
fits 32 GiB of VRAM without offload, serves a workload the suite measures,
and together the set spans two architecture families and both MoE and dense
designs — because a conclusion drawn from one vendor's models is a conclusion
about that vendor.

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
6. **Download `qwen3:30b`** (18.56 GB) and compare the official instruct
   build against the coder build on the same base.
7. **Optional offload experiment.** `gpt-oss:120b` (65.37 GB) or the 70B
   IQ3_XS tag, long-context category only, with a raised timeout, purely to
   quantify how badly offload hurts on this host.
8. **Suite `v2`, long-context category.** Stages 3 to 5 found the v1
   long-context tasks saturated: all five models scored 1.00 at every context
   length, including when 29% of the model was offloaded, so the category
   separated nothing. Suite `v2` replaces those three tasks with eight harder
   ones and inherits the other five categories from `v1` unchanged.
   - **8a**: all five measured models, `--suite v2 --category long_context`,
     3 repetitions, at the tasks' own `num_ctx` 16384. Estimated 15-25 minutes;
     the estimator costs this from v1 history, where the documents were the
     same size but the output budgets smaller, so treat it as a floor.
   - **8b**: the top two models again at `--context 32768`, which extends the
     stage 5 scaling curve onto tasks that can actually fail.

   Both are scripted: `./scripts/benchmark-stage.sh 8a`. **Both have now run,
   and the answer was negative**: v2 is saturated too, four of five models at
   1.00 on all eight tasks at both window sizes. See [RESULTS.md](RESULTS.md).
9. **The document-length sweep.** Stages 8a and 8b settled that task structure
   is not the binding constraint, and in the process established that document
   length had never been varied at all: every long-context measurement in this
   project used a document of roughly 8,000-10,000 tokens, because size lives
   in each task's `filler_tokens` and `--context` moves only the window. The
   `--filler` override exists as of 0.4.0 and these two stages use it.
   - **9a**: all five models, `--filler 16000 --context 32768`, 2 repetitions.
     Roughly double every document measured so far. Estimated 12-25 minutes;
     the estimator costs it from history at half the document size, so treat
     it as a floor — prompt evaluation is the part that grows.
   - **9b**: `gpt-oss:20b` and `qwen3-coder:30b` at `--filler 40000 --context
     65536`, roughly five times the measured document size. Only these two fit
     a 64k window: peak GPU memory at `num_ctx` 65536 measured 19,447 MiB and
     31,195 MiB of 32,760 total, so 9b is close to the card's limit and
     `qwen3-coder:30b` may partially offload, which the report records.

   Scripted as `./scripts/benchmark-stage.sh 9a`. Neither has run, and no
   result at any document size other than ~8,000 tokens exists yet. If models
   hold at 1.00 through 9b, the honest conclusion is that this hardware cannot
   hold a document large enough to challenge them, and the category should be
   reported as a pass/fail gate rather than a discriminator.

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
- An unrelated pre-existing GPU process has held the device throughout, using
  430 MiB. It is reported by preflight and never disturbed; its memory counts
  against available VRAM.
- Context capability varies sharply between tags on this host: some dense
  builds report 40960 tokens against the 262144 of the Qwen3 MoE builds. Do
  not compare long-context results across two models without checking the
  `model max context` column in the run report.
- vLLM 0.11.0 is installed in a separate conda environment on the benchmark
  host (Python 3.11.13, torch 2.8.0+cu128) and is not running. It cannot load Ollama's GGUF/MXFP4 artifacts directly, so a
  like-for-like comparison needs separately obtained weights. See
  [ROADMAP.md](ROADMAP.md).

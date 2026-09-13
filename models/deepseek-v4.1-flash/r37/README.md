# DS4.1 TP4/DCP1 artifact qualification

Status: qualified for the conditions recorded in `qualification.json`.
The repository-root model runbook is `models/deepseek-v4.1-flash.md`.

- `qualification.json`: artifact identities, pass/fail gates, medians, ranges,
  acceptance positions and the comparison boundary.
- `artifact-audit.json`: source trees, dependency hashes, native-binary parity
  and root-filesystem layer checks.
- `source.lock`: complete source and build-input manifest.
- `registry.json`: registry digest and verified identity after pulling it.
- `samples/`: uncached 32K prompts, two five-run Sieve series and both
  three-cell C1 series. `decode-respect-eos` uses default reasoning budget 75;
  `decode-budget50` injects numeric budget 50 through `chat_template_kwargs`.
- `control/`: public-source control on the R36 dependency runtime, not an
  execution of the unmodified R36 image. The control's default budget is 50.

The client uses llm-inference-bench source SHA-256
`516dc590b80dda6d9880f9f5034026fdf1b2074e9747bf8a920758680f019817`.
Only its self-updater is disabled. For explicit-budget cells, the HTTP adapter
adds `chat_template_kwargs.reasoning_effort=50` to chat request bodies without
changing benchmark sampling, request scheduling or measurement.

Physical GPUs 4–7 are RTX PRO 6000 Workstation cards with memory offset +6000,
graphics offset zero and automatic SM clocks. The Sieve receipts include GPU
UUIDs, offsets, clocks and temperatures. All comparisons are sequential and
stochastic; a measured total does not isolate each PR's effect.

The decode client's generic KV estimate is not authoritative for DS4.1's
heterogeneous native cache geometry. The 1,741,084-token pool in the summary
comes from the serving engine's initialized cache report.

The generated Sieve programs were not executed. These receipts establish
throughput and bounded smoke behavior, not program correctness, production
quality, SSD performance or long-context LMCache robustness.

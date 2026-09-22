# Configuration

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [MinusPod routing](#minuspod-routing)
- [Provider fields](#provider-fields)
- [Runtime threshold settings](#runtime-threshold-settings)
- [Category classification](#category-classification)
- [Review behavior](#review-behavior)
- [Sponsor naming](#sponsor-naming)
- [Runtime ownership](#runtime-ownership)

## MinusPod routing

MinusPod routes a provider per phase in `llm_route.py`:

1. Set detection and verification to the proxy.
2. For independent review and chapter titles, configure a separate chat model as the secondary provider.

This is a recommended POC configuration, not a change applied to MinusPod by this repository.

> Note: Jev Proxy does not support chapter generation. You still need a chat model for chapter titles. Configure a secondary chat-model provider and set `chapters_provider` to `secondary`; do not route chapter generation to this proxy.

## Provider fields

| field | value |
|---|---|
| provider | `openai_compatible` |
| base URL | your proxy's base URL |
| model | `typesafe/jev` |
| addressing mode | `timestamps` |
| API key | your **TypeSafe key**, forwarded by the proxy to Jev |

| phase | provider slot | goes to |
|---|---|---|
| `detection_provider` | primary | proxy -> Jev |
| `verification_provider` | primary | proxy -> Jev |
| `review_provider` | secondary | independent chat model |
| `chapters_provider` | secondary | a real chat model |

### Policy and limits

- The caller's system policy is forwarded unchanged to Jev's detection and category classification guidance.
- This shim does not impose text, segment, or request-body caps and never truncates the prompt; Jev enforces its current model limits upstream.
- External network components may enforce their own limits outside this shim. See the [Jev model limits](https://docs.typesafe.ai/models) and [API documentation](https://docs.typesafe.ai/api) for the current upstream contract.

## Runtime threshold settings

| setting | runtime field | default | purpose |
|---|---|---|---|
| `JEV_ENTER` | `detection_enter` | `0.95` | opens an ad span; editable at runtime |
| `JEV_STAY` | `detection_stay` | `0.40` | extends an open span; editable at runtime |
| `JEV_REVIEW_EVIDENCE_THRESHOLD` | `review_evidence` | `JEV_ENTER` when unset | gates advertising evidence before refinement |
| `JEV_REVIEW_CHOICE_THRESHOLD` | `review_choice` | `JEV_ENTER` when unset | gates the selected boundary pair |

These are `0` to `1` probability scores, not measured accuracy. Detection enter must be at least `JEV_STAY`; review evidence and Choice thresholds are independent of detection enter.

### Save and authentication

- The status page reads `GET /api/settings` and saves all four editable thresholds atomically with `PUT /api/settings`.
- Saving requires `Authorization: Bearer <MinusPod password>`. The proxy checks that password locally and does not log in to MinusPod. The password is not saved in browser storage.
- Without `MINUSPOD_PASSWORD`, settings are visible but not editable.

### Persistence and request scope

- Changes apply to new requests. In-flight requests keep their existing snapshot, so avoid changing thresholds during an episode if its windows must use one policy.
- Saved overrides use `JEV_SETTINGS_PATH` (default `./data/runtime-settings.json`) and survive restarts when that directory is persistent.
- A saved four-field file overrides environment defaults until it is changed or removed.
- A legacy three-field file uses the startup `JEV_STAY` value until its next save writes all four fields.
- Corrupt or unreadable state returns 503 rather than silently resetting to defaults. The Compose file mounts `/app/data` to a named volume. Existing deployments must add an equivalent persistent mount; replacing only the image does not preserve the settings file.

### Troubleshooting

- If either settings endpoint returns 503, inspect proxy logs for `runtime settings storage failure operation=read|write errno=<number>`.
- The message omits paths and file contents.
- The `/app/data` mount must be writable by UID/GID 1000.
- Existing volume ownership overrides image defaults, so verify the mounted directory before applying a nonrecursive `chown` to UID/GID 1000.

## Category classification

The category pass uses one Jev Choice for each detected span. Its values are `sponsor`, `cross_promo`, `self_promo`, `interaction`, `intro`, `outro`, and `recap`.

- `JEV_CATEGORY_PASS=true` enables the pass. With `false`, the proxy uses `JEV_DEFAULT_CATEGORY`.
- Category Choice has no threshold. It does not use the detection or boundary-review thresholds.
- A category result does not change the detected span confidence.

## Review behavior

- Jev review is correlated with Jev detection, not an independent judgment. It uses coarse context and separate word timing.
- The evidence NouL adds an upstream call unless cached.
- `JEV_REVIEW_REFINE_BOUNDARIES=false` disables boundary-pair refinement by default. Set it to `true` to select one complete range from supplied word timings.
- A pair can retain the original boundary or use the closest meaningful timed boundary: a sentence break or a coarse segment edge with an exact word timestamp. It must overlap both the candidate and Jev's corroborated span.
- The Choice contains up to 9 valid combinations of keep, trim, or extend boundaries, plus unknown.
- `GET /api/status` reports effective enabled state, model, evidence threshold, and Choice threshold. Enabled does not guarantee refinement: word timings, sufficient evidence, and a confident pair selection are required.
- This local candidate set can miss further or intrasentence corrections. It does not claim an efficacy improvement.
- Review logs include request ID, stage, evidence score and threshold, word counts, Choice confidence, and skip or failure reason.
- An unknown or low-confidence pair selection returns 422 with `x-should-retry: false` and does not confirm or move the candidate.
- MinusPod's local breaker still counts non-rate errors. Review preserves upstream `4xx` responses, including `429`. A valid `Retry-After` header is forwarded for upstream `408`, `429`, and `5xx` responses. Timeouts return `504`; transport and upstream `5xx` failures return `503`. Invalid review responses return `503`.

## Sponsor naming

- Set `MINUSPOD_BASE_URL` and `MINUSPOD_PASSWORD`; the proxy logs into MinusPod (cached session) to read `GET /api/v1/sponsors`.
- It emits `sponsor_name` only for one known sponsor with local ad evidence, using the `jev-` namespace, for example `jev-ButcherBox`.
- The prefix identifies a proxy-generated learned record and the suffix is the matched canonical brand. An unmatched span leaves the field absent, preventing false sponsor evidence.
- Its `Based on transcript:` rationale quotes source evidence rather than synthetic ad wording, and the compatibility parser recognizes it as rationale.
- Unset `MINUSPOD_BASE_URL` falls back to the gazetteer.

## Runtime ownership

Jev Proxy is a POC shim. It does not change the MinusPod runtime that controls holds, autoapproval, or verification logs. `compat/minuspod_compat` is a vendored snapshot used by the production proxy and offline benchmarks. Keep production operations in the existing MinusPod application until Jev is mature.

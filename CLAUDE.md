# Working on forced-aligner

Read [README.md](README.md) first for what this is. This file is how to work
on it.

This is the alignment service for
[the sing thing](https://github.com/stevelittlefish/the_sing_thing). It exists
so that project can stay pure Go: all the ugly ML (PyTorch, wav2vec2, CUDA)
lives here, behind an HTTP boundary, and the karaoke app just makes a request.

## Raw dog commit to main YOLO

The commit policy. **Commit everything straight to `main`, and push every time
you commit.** No branches, no PRs, no waiting to be asked. Finish a coherent
chunk, run the checks, commit with a message explaining *why*, `git push`, move
on. The push is the YOLO — it is not optional.

Append a Claude Code-style co-author trailer for the model in use, e.g.
`Co-authored-by: GPT 6 Astra <noreply@openai.com>` for that Codex model.

Corrections are follow-up commits, not history rewrites: got it wrong, commit
the fix on top and say what it fixes. Once it's in, it stays in.

## Fixed choices

- **Python 3.11+.** This is where Python is *allowed* — it's the reason the
  service exists. `tomllib` is stdlib from 3.11, which is why config needs no
  dependency.
- **No environment variables. Configuration is `config.toml`, always.** Read
  with stdlib `tomllib`. Same rule as the sing thing — services on this LAN are
  configured by file, not environment.
- **FastAPI + uvicorn** for the HTTP surface. **WhisperX** for alignment;
  **PyTorch** underneath. These are the real dependencies and they justify
  themselves — the whole job is running a wav2vec2 model on a GPU.
- **ASS job contract.** `/v1/align`, job polling, artifact downloads, `/health`
  and `/v1/info`. Synchronous `/align` is restored for standalone callers;
  it shares the job worker and limits, but returns JSON directly and cleans up
  its scratch files. `/models` remains replaced by `/v1/info`.
  Use one serial inference worker and one resident language model. `/park` and
  `/unpark` are implemented (move the resident model to CPU RAM and back, freeing
  the GPU), so ASS can evict with `park` and skip our cold start. Stop eviction
  still works for backends configured that way.
- **Multipart in, asynchronous jobs out.** Save timings as `alignment.json`;
  ASS harvests it before removing the container. No automatic result expiry.
- The image supplies TOML defaults. HF/torch cache environment variables are
  allowed as library wiring; all paths are under `/cache/aligner`, with the
  shared token at `/cache/hf-token`.

## Layout

| | |
|---|---|
| `app/main.py` | FastAPI app: routes, request/response shapes, model cache |
| `app/align.py` | The alignment core — load audio, run WhisperX, map words to lines |
| `app/config.py` | Loads `config.toml` via `tomllib` |
| `docs/spec.md` | The API contract — request/response, error cases |
| `docs/alignment.md` | How CTC forced alignment works and where it struggles |
| `docs/deploy.md` | Running it as a container on the AI server |
| `config.example.toml` | Copy to `config.toml` and edit |

## The alignment core follows nightingale

`app/align.py` is modelled on nightingale's
`app-core/analyzer/align.py` (studied, not copied — different language, different
lifecycle). The approach: VAD-trim the vocals, feed the whole known lyric as one
segment, run wav2vec2 CTC forced alignment, group characters into words, map
words back onto the original lines. When in doubt about *how to align well*,
that file is the reference.

## Checks before committing

```sh
ruff check . && python -m compileall app && pytest -q
```

Tests exist now (`tests/`, no GPU/ML stack needed — the aligner is mocked).
Install them with `pip install -r requirements-dev.txt`. A chunk that fails its
checks is not finished.

## Honesty about singing

wav2vec2 is trained on speech, not singing. Sustained notes smear word-ends,
harmonies confuse it, screams and poor enunciation degrade it. **Do not paper
over this** — surface a confidence/failure signal in the response (unresolved
words get `null` timestamps rather than a confident lie) so the caller can fall
back to line-level timing or flag the song for manual correction. Getting a
clear "this one is bad" out is as valuable as getting good timings.

## Writing

Plain, direct, willing to say when something is unresolved or wrong. No
marketing tone, no hedging. Comments explain **why**, never what. When a later
decision overturns an earlier one, say so in the earlier place with a pointer
rather than quietly rewriting it.

# Academic Distribution Notes

This repository is submitted to the faculty as supporting source code for an
academic thesis. It is not published under an open-source license by this
submission.

## Permitted Scope

The source-code archive is intended for thesis examination, academic review,
and institutional archiving. Copyright in the original project source remains
with the author unless a separate written agreement states otherwise.

No permission for commercial redistribution, sublicensing, or republication is
granted implicitly. A future public release should add an explicit software
license selected by the author.

## Third-Party Materials

- Qwen model files are not included. Their use remains subject to the license
  and terms published by the respective model provider.
- BPS publications, raw source documents, and substantial derived corpora are
  not included in the faculty source-code ZIP. Their copyright and reproduction
  conditions remain with Badan Pusat Statistik and the relevant publishers.
- Python packages remain subject to their individual licenses.
- Minimal question/answer ground truth and aggregate evaluation artifacts are
  included solely to explain and verify the thesis implementation.

## Excluded Runtime Data

The archive builder excludes model weights, embeddings, ChromaDB state, raw and
cleaned publications, generated chunk corpora, chat history, backups, caches,
logs, local IDE/agent configuration, Git history, and secrets.

See `docs/DEVELOPER_HANDOFF.md` for the exact archive contents and setup steps.

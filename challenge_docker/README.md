# Directory-mounted policy runtime

The image (built with `challenge_docker/Dockerfile`, see the repository README) contains the code;
the shared third-party inference assets are bind-mounted at `/opt/fastwam/runtime-assets`.  Each model is
provided as a separate read-only directory with this layout:

```text
model_a/
  checkpoint.pt
  checkpoint_contract.json
  norm_stats.json
  model_config.json
```

The entrypoint reads the directory and derives the supported configuration, so
model-specific `-e` variables are not needed. `model_config.json` enables task
and batch guards; if it is absent, the previous unrestricted behavior is kept.

```bash
docker run --rm --name policy-runtime \
  --gpus device=0 --network host \
  -v /path/model_a:/runtime/model:ro \
  fastwam-policy-runtime --host 0.0.0.0 --port 8000
```

Before allocating a GPU, validate a root containing several model directories:

```bash
python /opt/fastwam/challenge_docker/preflight_models.py /path/models
```

The delivered contract has empty provenance. Compatibility fields and the
normalization statistics remain integrity checked at startup.

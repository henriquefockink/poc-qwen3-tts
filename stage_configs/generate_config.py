#!/usr/bin/env python3
"""Generate stage_configs YAML from environment variables.

Reads the base qwen3_tts.yaml template and overrides values based on
environment variables, writing the result to /tmp/stage_configs.yaml.
"""

import os
import yaml

BASE_CONFIG = os.path.join(os.path.dirname(__file__), "qwen3_tts.yaml")
OUTPUT_PATH = "/tmp/stage_configs.yaml"


def env(name: str, default=None, cast=None):
    """Read an env var with optional type casting."""
    val = os.environ.get(name, default)
    if val is None:
        return None
    if cast is not None:
        if cast is bool:
            return str(val).lower() in ("true", "1", "yes")
        return cast(val)
    return val


def main():
    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)

    stage0 = config["stage_args"][0]
    stage1 = config["stage_args"][1]

    # --- Stage 0 (Talker) engine_args ---
    s0_engine = stage0["engine_args"]
    s0_engine["max_num_seqs"] = env("MAX_NUM_SEQS", s0_engine["max_num_seqs"], int)
    s0_engine["gpu_memory_utilization"] = env(
        "GPU_MEMORY_UTILIZATION_STAGE0", s0_engine["gpu_memory_utilization"], float
    )
    s0_engine["max_model_len"] = env(
        "MAX_MODEL_LEN_STAGE0", s0_engine["max_model_len"], int
    )
    s0_engine["enforce_eager"] = env(
        "ENFORCE_EAGER", s0_engine["enforce_eager"], bool
    )
    s0_engine["trust_remote_code"] = env(
        "TRUST_REMOTE_CODE", s0_engine["trust_remote_code"], bool
    )

    # --- Stage 0 sampling params ---
    s0_sampling = stage0["default_sampling_params"]
    s0_sampling["temperature"] = env("TEMPERATURE", s0_sampling["temperature"], float)
    s0_sampling["top_k"] = env("TOP_K", s0_sampling["top_k"], int)
    s0_sampling["top_p"] = env("TOP_P", s0_sampling.get("top_p"), float)
    s0_sampling["max_tokens"] = env("MAX_TOKENS", s0_sampling["max_tokens"], int)
    s0_sampling["repetition_penalty"] = env(
        "REPETITION_PENALTY", s0_sampling["repetition_penalty"], float
    )
    s0_sampling["seed"] = env("SEED", s0_sampling["seed"], int)

    # Add top_p only if set (not in default stage0 config)
    if s0_sampling.get("top_p") is not None:
        s0_sampling["top_p"] = s0_sampling["top_p"]
    else:
        s0_sampling.pop("top_p", None)

    # --- Stage 1 (Code2Wav) engine_args ---
    s1_engine = stage1["engine_args"]
    s1_engine["gpu_memory_utilization"] = env(
        "GPU_MEMORY_UTILIZATION_STAGE1", s1_engine["gpu_memory_utilization"], float
    )
    s1_engine["max_model_len"] = env(
        "MAX_MODEL_LEN_STAGE1", s1_engine["max_model_len"], int
    )
    s1_engine["enforce_eager"] = env(
        "ENFORCE_EAGER", s1_engine["enforce_eager"], bool
    )
    s1_engine["trust_remote_code"] = env(
        "TRUST_REMOTE_CODE", s1_engine["trust_remote_code"], bool
    )

    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Stage config written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

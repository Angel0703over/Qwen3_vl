"""KV cache helpers for Qwen3-VL runtime-only generate."""

from .kv_cache import (
    LayerKVCache,
    StageKVCache,
    build_stage_kv_cache,
)
from .video_kv_compression import (
    VIDEO_KV_COMPACTION_SCHEMA,
    VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA,
    VIDEO_KV_COMPRESSION_METHODS,
    VIDEO_KV_COMPRESSION_PLAN_SCHEMA,
    VIDEO_KV_SELECTOR_METHODS,
    build_compact_decode_attention_mask_2d,
    build_compact_prefill_attention_mask_2d,
    build_video_kv_compression_contract,
    build_video_kv_compression_plan,
    compact_stage_kv_cache_for_video_plan,
    materialize_video_kv_compression_plan,
    resolve_prefill_keep_token_indices,
    validate_video_kv_compression_decode_contract,
)
from .video_window_cache import (
    KVLocation,
    VIDEO_WINDOW_CACHE_SCHEMA,
    VideoWindowCacheIndex,
    VideoWindowId,
    VideoWindowMetadata,
    attach_video_window_cache_index,
    build_video_window_cache_index,
)

__all__ = [
    "LayerKVCache",
    "StageKVCache",
    "build_stage_kv_cache",
    "KVLocation",
    "VIDEO_WINDOW_CACHE_SCHEMA",
    "VideoWindowCacheIndex",
    "VideoWindowId",
    "VideoWindowMetadata",
    "attach_video_window_cache_index",
    "build_video_window_cache_index",
    "VIDEO_KV_COMPACTION_SCHEMA",
    "VIDEO_KV_COMPRESSION_CONTRACT_SCHEMA",
    "VIDEO_KV_COMPRESSION_METHODS",
    "VIDEO_KV_COMPRESSION_PLAN_SCHEMA",
    "VIDEO_KV_SELECTOR_METHODS",
    "build_compact_decode_attention_mask_2d",
    "build_compact_prefill_attention_mask_2d",
    "compact_stage_kv_cache_for_video_plan",
    "materialize_video_kv_compression_plan",
    "build_video_kv_compression_contract",
    "build_video_kv_compression_plan",
    "resolve_prefill_keep_token_indices",
    "validate_video_kv_compression_decode_contract",
]

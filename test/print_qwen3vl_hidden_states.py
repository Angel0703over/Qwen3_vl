import os
import glob

import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.masking_utils import create_causal_mask
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/mnt/ssd/models/Qwen/Qwen3-VL-4B-Instruct"
FRAME_DIR = "/mnt/ssd/code/Qwen3_vl/frames"

# 跑完第 k 层停下，然后从第 k+1 层继续
CUT_LAYER_IDX = 11
NUM_FRAMES = 8


class StopForward(Exception):
    pass


def find_decoder_layers(model: nn.Module, expected_num_layers: int = 36):
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == expected_num_layers:
            types = [type(m).__name__ for m in module[: min(3, len(module))]]
            candidates.append((name, module, types))

    if not candidates:
        raise RuntimeError(
            f"没有找到长度为 {expected_num_layers} 的 ModuleList，请手动检查模型结构。"
        )

    print("\n===== MODULELIST CANDIDATES =====")
    for name, _, types in candidates:
        print(f"{name} -> {types}")

    chosen_name, chosen_module, _ = candidates[0]
    print(f"\nUsing decoder layers: {chosen_name}")
    return chosen_name, chosen_module


def build_inputs(processor, frame_paths):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [f"file://{p}" for p in frame_paths],
                    "sample_fps": 1,
                },
                {
                    "type": "text",
                    "text": "请用中文简要描述这个视频的主要内容。",
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )

    # 某些版本里会出现这个键，保险起见去掉
    inputs.pop("token_type_ids", None)

    return inputs


def print_input_summary(inputs):
    print("\n===== INPUT KEYS =====")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"{k}: {type(v)}")

def compute_position_ids(model, inputs):
    return model.model.compute_3d_position_ids(
        input_ids=inputs["input_ids"],
        inputs_embeds=model.model.get_input_embeddings()(inputs["input_ids"]),
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs.get("attention_mask"),
        past_key_values=None,
        mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )


def compute_remaining_deepstack_inputs(model, inputs, cut_layer_idx: int):
    num_deepstack_layers = len(model.model.visual.deepstack_visual_indexes)
    if cut_layer_idx >= num_deepstack_layers - 1:
        return None, None

    input_ids = inputs["input_ids"]
    input_embeds = model.model.get_input_embeddings()(input_ids)
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")

    image_mask = None
    video_mask = None
    deepstack_image_embeds = None
    deepstack_video_embeds = None

    if inputs.get("pixel_values") is not None:
        image_outputs = model.model.get_image_features(
            inputs["pixel_values"],
            image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(input_embeds.device, input_embeds.dtype)
        deepstack_image_embeds = image_outputs.deepstack_features
        image_mask, _ = model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=input_embeds,
            image_features=image_embeds,
        )

    if inputs.get("pixel_values_videos") is not None:
        video_outputs = model.model.get_video_features(
            inputs["pixel_values_videos"],
            video_grid_thw,
            return_dict=True,
        )
        video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(input_embeds.device, input_embeds.dtype)
        deepstack_video_embeds = video_outputs.deepstack_features
        _, video_mask = model.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=input_embeds,
            video_features=video_embeds,
        )

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        visual_pos_masks = video_mask[..., 0]
        deepstack_visual_embeds = deepstack_video_embeds

    return visual_pos_masks, deepstack_visual_embeds


def resume_decoder_from_layer(model, cut_hidden, inputs, cut_layer_idx: int):
    text_model = model.model.language_model
    position_ids = compute_position_ids(model, inputs)
    attention_mask = create_causal_mask(
        config=text_model.config,
        inputs_embeds=cut_hidden,
        attention_mask=inputs.get("attention_mask"),
        past_key_values=None,
        position_ids=None,
    )
    position_embeddings = text_model.rotary_emb(cut_hidden, position_ids)
    visual_pos_masks, deepstack_visual_embeds = compute_remaining_deepstack_inputs(model, inputs, cut_layer_idx)

    hidden_states = cut_hidden
    for layer_idx in range(cut_layer_idx + 1, len(text_model.layers)):
        hidden_states = text_model.layers[layer_idx](
            hidden_states,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            position_embeddings=position_embeddings,
        )
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden_states = text_model._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    return text_model.norm(hidden_states)


def main():
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    assert frame_paths, f"No frames found in {FRAME_DIR}"
    frame_paths = frame_paths[:NUM_FRAMES]
    print(f"Using {len(frame_paths)} frames")

    print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
    ).eval()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    inputs = build_inputs(processor, frame_paths)
    inputs = inputs.to(model.device)

    print_input_summary(inputs)

    # ===== 1) 完整 forward，拿参考 logits =====
    print("\nRunning full forward...")
    with torch.inference_mode():
        full_outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    full_logits = full_outputs.logits
    hidden_states = full_outputs.hidden_states

    print("\n===== 完整前向结果摘要 =====")
    print("hidden_states 个数：", len(hidden_states))
    print("完整 logits 形状：", tuple(full_logits.shape))

    # 自动找 decoder layer
    _, layers = find_decoder_layers(model, expected_num_layers=len(hidden_states) - 1)

    if not (0 <= CUT_LAYER_IDX < len(layers)):
        raise ValueError(
            f"CUT_LAYER_IDX={CUT_LAYER_IDX} 越界，当前 decoder 层数为 {len(layers)}"
        )

    # ===== 2) 截断到第 k 层，拿边界 hidden state =====
    captured = {}

    def stop_hook(module, module_inputs, module_output):
        if isinstance(module_output, tuple):
            hidden = module_output[0]
        else:
            hidden = module_output
        captured["hidden"] = hidden.detach().clone()
        raise StopForward

    handle = layers[CUT_LAYER_IDX].register_forward_hook(stop_hook)

    print(f"\n正在执行截断前向（跑到第 {CUT_LAYER_IDX} 层后停止）……")
    try:
        with torch.inference_mode():
            _ = model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
    except StopForward:
        print(f"Stopped after layer {CUT_LAYER_IDX}")
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError("没有捕获到 cut_hidden，请检查 hook。")

    cut_hidden = captured["hidden"]
    ref_hidden = hidden_states[CUT_LAYER_IDX + 1]

    print("\n===== 截断正确性检查 =====")
    print("截断 hidden 形状：", tuple(cut_hidden.shape))
    print("参考 hidden 形状：", tuple(ref_hidden.shape))
    print("截断 hidden 与参考 hidden 的最大绝对误差：", (cut_hidden - ref_hidden).abs().max().item())
    print("截断 hidden 与参考 hidden 的平均绝对误差：", (cut_hidden - ref_hidden).abs().mean().item())

    # ===== 3) 从第 k+1 层继续跑 =====
    # 思路：直接调用 language_model，并把前半段 layer 临时替换成 identity
    # 这样输入 cut_hidden 后，真正生效的只有后半段层
    print(f"\nRunning resumed forward from layer {CUT_LAYER_IDX + 1} ...")

    with torch.inference_mode():
        resumed_last_hidden = resume_decoder_from_layer(model, cut_hidden, inputs, CUT_LAYER_IDX).detach().clone()
        resumed_logits = model.lm_head(resumed_last_hidden)

    # ===== 4) 和完整 forward 的 logits 对比 =====
    print("\n===== 恢复执行正确性检查 =====")
    print("恢复后 last_hidden_state 形状：", tuple(resumed_last_hidden.shape))
    print("恢复后 logits 形状：", tuple(resumed_logits.shape))
    print("完整 logits 形状：", tuple(full_logits.shape))

    logits_max_abs_diff = (resumed_logits - full_logits).abs().max().item()
    logits_mean_abs_diff = (resumed_logits - full_logits).abs().mean().item()

    print("resumed vs full logits max abs diff:", logits_max_abs_diff)
    print("resumed vs full logits mean abs diff:", logits_mean_abs_diff)
    
    print("\n===== 最后一个 token 的 logits 前 8 维 =====")
    print("恢复执行结果：", resumed_logits[0, -1, :8].float().cpu())
    print("完整前向结果：", full_logits[0, -1, :8].float().cpu())

    print("\n全部完成。")


if __name__ == "__main__":
    main()

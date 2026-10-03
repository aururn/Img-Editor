"""Gradio Blocks definition for Img Editor."""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from ..core import AppConfig
from . import handlers
from .handlers import HandlerContext

MODES = [
    handlers.TXT2IMG,
    handlers.IMG2IMG,
    handlers.INPAINT_MANUAL,
    handlers.INPAINT_AUTO,
]
SAMPLERS = ["Euler a", "DPM++ 2M Karras", "DDIM"]
REGIONAL_LAYOUTS = ["horizontal", "vertical", "grid", "mask"]
CONTROL_PREPROCESS = ["none", "canny"]


CUSTOM_CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")


def build_ui(config: AppConfig) -> gr.Blocks:
    ctx: HandlerContext = handlers.build_context(config)
    defaults = config.defaults or {}
    inpaint_defaults = defaults.get("inpaint", {})
    model_name = config.model.get("base_checkpoint", "—")

    with gr.Blocks(title="Img Editor", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:

        # ── Topbar ───────────────────────────────────────────────────
        with gr.Row(elem_id="topbar"):
            gr.HTML(
                "<div class='topbar-inner'>"
                "<div class='topbar-logo'>E</div>"
                "<span class='topbar-title'>Img Editor</span>"
                "<span class='topbar-subtitle'>SDXL · LoRA · Regional</span>"
                f"<span class='topbar-model'>{model_name}</span>"
                "</div>"
            )

        # ── System Status (collapsed by default) ─────────────────────
        with gr.Accordion("System Status", open=False, elem_id="diag-accordion"):
            gr.Markdown(handlers.diagnostics_markdown(ctx))

        # ── Mode selector (tab-bar style) ─────────────────────────────
        mode = gr.Radio(
            MODES,
            value=handlers.INPAINT_AUTO,
            label="",
            show_label=False,
            elem_id="mode-selector",
        )

        # ── 3-column main area ────────────────────────────────────────
        with gr.Row(elem_id="main-row"):

            # ─── LEFT RAIL: LoRA + Tool Accordions ───────────────────
            with gr.Column(scale=3, min_width=260, elem_id="left-rail"):

                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Checkpoint</div>')
                    checkpoint = gr.Dropdown(
                        choices=handlers.checkpoint_choices(ctx),
                        value=handlers.default_checkpoint(ctx),
                        label="Checkpoint",
                        elem_id="checkpoint-dropdown",
                    )
                    refresh_checkpoints = gr.Button("Reload checkpoints", size="sm")

                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">LoRA</div>')
                    lora_search = gr.Textbox(
                        placeholder="Search: name · trigger · warning",
                        show_label=False,
                    )
                    with gr.Row():
                        refresh_loras = gr.Button("Reload", size="sm")
                        insert_lora_btn = gr.Button("Insert triggers", size="sm")
                    lora_table = gr.CheckboxGroup(
                        choices=handlers.lora_choices(ctx),
                        label=None,
                        show_label=False,
                        interactive=True,
                        elem_id="lora-checkboxes",
                    )
                    lora_upload = gr.File(
                        label="Upload LoRA (.safetensors)",
                        file_types=[".safetensors"],
                        file_count="multiple",
                        type="filepath",
                    )

            # ─── CENTER: Canvas + Prompt + Parameters + Generate ──────
            with gr.Column(scale=5, min_width=460, elem_id="center-panel"):

                img2img_image = gr.Image(
                    label="Input image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    visible=False,
                    elem_id="img2img-box",
                )
                inpaint_canvas = gr.ImageEditor(
                    label="Canvas",
                    type="pil",
                    brush=gr.Brush(
                        colors=["#ff0000", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7"],
                        default_size=24,
                    ),
                    sources=["upload", "clipboard"],
                    visible=True,
                    elem_id="canvas-box",
                )
                canvas_state = gr.State(value=None)
                inpaint_canvas.change(
                    lambda v: v, inputs=inpaint_canvas, outputs=canvas_state
                )

                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Prompt</div>')
                    prompt = gr.Textbox(
                        show_label=False,
                        lines=5,
                        placeholder="masterpiece, best quality, 1girl, empty eyes",
                        elem_id="prompt-box",
                    )
                with gr.Accordion("Negative Prompt", open=False):
                    negative_prompt = gr.Textbox(
                        show_label=False,
                        lines=2,
                        value="lowres, bad anatomy, blurry",
                    )

                with gr.Accordion("Regional Prompter", open=False):
                    regional_enabled = gr.Checkbox(label="Enable regional prompt", value=False)
                    regional_layout = gr.Dropdown(
                        REGIONAL_LAYOUTS, value="horizontal", label="Layout"
                    )
                    regional_ratios = gr.Textbox(
                        label="Ratios", placeholder="1,1 or 1,2,1;1,1"
                    )
                    regional_base_ratios = gr.Textbox(
                        label="Base ratio",
                        placeholder="0.2 or 0.2,0.3 (blank = 0.2)",
                    )
                    regional_overlay_ratio = gr.Slider(
                        0.0,
                        0.5,
                        value=0.0,
                        step=0.01,
                        label="Overlay ratio",
                    )
                    with gr.Row():
                        regional_use_base_prompt = gr.Checkbox(
                            label="Use base prompt",
                            value=False,
                        )
                        regional_use_common_prompt = gr.Checkbox(
                            label="Use common prompt",
                            value=False,
                        )
                    regional_use_common_negative = gr.Checkbox(
                        label="Use common negative prompt",
                        value=True,
                    )
                    regional_common_prompt = gr.Textbox(
                        label="Common prompt", lines=2,
                        placeholder="tags shared by all regions, e.g. 2girls, classroom",
                    )
                    regional_prompt_text = gr.Textbox(
                        label="Region prompts", lines=4,
                        placeholder="left region prompt\nBREAK\nright region prompt",
                    )
                    regional_lora_text = gr.Textbox(
                        label="Region LoRAs", lines=3,
                        placeholder="<lora:left_character:0.8>\nBREAK\n<lora:right_character:0.8>",
                    )
                    with gr.Row():
                        regional_lora_negative_te = gr.Textbox(
                            label="LoRA negative TE",
                            placeholder="0 or 0,0.2",
                        )
                        regional_lora_negative_unet = gr.Textbox(
                            label="LoRA negative U-Net",
                            placeholder="0 or 0,0.2",
                        )
                    regional_lora_stop_step = gr.Number(
                        value=0,
                        precision=0,
                        label="LoRA stop step",
                    )
                    regional_negative_text = gr.Textbox(
                        label="Region negatives", lines=2,
                        placeholder="optional, separated by BREAK",
                    )
                    regional_preview_btn = gr.Button("Preview masks", size="sm")
                    regional_mask_preview = gr.Image(
                        label="Mask preview",
                        type="pil",
                        interactive=False,
                        height=180,
                    )
                    gr.Markdown(
                        "Supports BREAK, ADDROW, ADDCOL, ADDBASE, ADDCOMM, "
                        "base ratios, per-region negatives, and per-region LoRAs. "
                        "Inline <lora:name:weight> tags are stripped from prompts and loaded as adapters.",
                    )

                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Basic Settings</div>')
                    with gr.Row():
                        cfg_scale = gr.Slider(1.0, 20.0, value=7.0, step=0.1, label="CFG")
                        steps = gr.Slider(10, 80, value=28, step=1, label="Steps")
                    with gr.Row():
                        width = gr.Slider(
                            512, 1536, value=1024, step=64, label="Width"
                        )
                        height = gr.Slider(
                            512, 1536, value=1024, step=64, label="Height"
                        )
                    with gr.Row():
                        sampler = gr.Dropdown(
                            SAMPLERS,
                            value=config.model.get("default_sampler", "DPM++ 2M Karras"),
                            label="Sampler",
                            elem_id="sampler-dropdown",
                        )
                    with gr.Row():
                        seed = gr.Number(
                            value=-1, precision=0, label="Seed",
                            elem_classes="seed-number", scale=2,
                        )
                        randomize = gr.Button("⟳", scale=0, size="sm")

                settings_summary = gr.Markdown(
                    "CFG **7.0** · Steps **28** · **1024×1024** · DPM++ 2M Karras",
                    elem_classes="settings-summary",
                )

                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate",
                        variant="primary",
                        elem_classes="primary-run",
                        scale=3,
                    )
                    cancel_btn = gr.Button("Cancel", scale=1)

            # ─── RIGHT: Result + History + Presets ───────────────────
            with gr.Column(scale=4, min_width=300, elem_id="right-panel"):

                with gr.Group(elem_classes="panel"):
                    gr.HTML('<div class="section-title">Result</div>')
                    result_image = gr.Image(
                        label=None,
                        show_label=False,
                        type="pil",
                        interactive=False,
                        elem_id="result-image-box",
                    )
                    result_info = gr.Markdown("", elem_classes="result-meta")
                    with gr.Row():
                        used_seed = gr.Number(
                            label="Seed used",
                            interactive=False,
                            precision=0,
                            elem_classes="seed-number",
                            scale=1,
                        )
                    with gr.Row():
                        reedit_img2img = gr.Button(
                            "→ img2img", size="sm", elem_classes="send-btn"
                        )
                        reedit_inpaint = gr.Button(
                            "→ inpaint", size="sm", elem_classes="send-btn"
                        )

                with gr.Accordion("History", open=True):
                    history_gallery = gr.Gallery(
                        label=None,
                        show_label=False,
                        columns=4,
                        height=160,
                        allow_preview=True,
                        object_fit="cover",
                    )

                with gr.Accordion("Presets", open=False):
                    with gr.Row():
                        preset_name = gr.Textbox(label="Name", scale=2)
                        preset_save = gr.Button("Save", scale=1, size="sm")
                    with gr.Row():
                        preset_select = gr.Dropdown(
                            choices=handlers.list_presets_choices(ctx),
                            label="Preset",
                            scale=2,
                        )
                        preset_load = gr.Button("Load", scale=1, size="sm")

        # ─── ADVANCED SETTINGS (below main row) ──────────────────────
        with gr.Accordion("Advanced Settings", open=False):
            with gr.Row():
                strength = gr.Slider(
                    0.0, 1.0, value=0.55, step=0.01, label="Strength"
                )
                mask_blur = gr.Slider(
                    0, 32,
                    value=int(inpaint_defaults.get("mask_blur", 8)),
                    step=1,
                    label="Mask blur",
                )

            with gr.Accordion("Textual Inversion", open=False):
                embedding_search = gr.Textbox(placeholder="Search embeddings", show_label=False)
                embeddings = gr.CheckboxGroup(
                    choices=handlers.embedding_choices(ctx),
                    label=None,
                )
                insert_embedding_btn = gr.Button("Insert embedding tokens", size="sm")
                embedding_upload = gr.File(
                    label="Upload embedding (.safetensors / .pt / .bin)",
                    file_types=[".safetensors", ".pt", ".bin"],
                    file_count="multiple",
                    type="filepath",
                )

            with gr.Accordion("ControlNet", open=False):
                controlnet_model = gr.Dropdown(
                    choices=handlers.controlnet_choices(ctx),
                    value="",
                    label="Model",
                )
                controlnet_upload = gr.File(
                    label="Upload ControlNet",
                    file_types=[".safetensors", ".bin"],
                    file_count="multiple",
                    type="filepath",
                )
                controlnet_image = gr.Image(
                    label="Control image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                controlnet_preprocess = gr.Dropdown(
                    CONTROL_PREPROCESS,
                    value=config.raw.get("controlnet", {}).get("preprocess", "none"),
                    label="Preprocess",
                )
                controlnet_scale = gr.Slider(
                    0.0, 2.0,
                    value=float(config.raw.get("controlnet", {}).get("scale", 1.0)),
                    step=0.05,
                    label="Scale",
                )
                with gr.Row():
                    controlnet_start = gr.Slider(
                        0.0, 1.0, value=0.0, step=0.05, label="Start"
                    )
                    controlnet_end = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.05, label="End"
                    )

            with gr.Accordion("IP-Adapter", open=False):
                ip_adapter_name = gr.Dropdown(
                    choices=handlers.ip_adapter_choices(ctx),
                    value="",
                    label="Weights",
                )
                ip_adapter_upload = gr.File(
                    label="Upload IP-Adapter",
                    file_types=[".safetensors", ".bin"],
                    file_count="multiple",
                    type="filepath",
                )
                ip_adapter_image = gr.Image(
                    label="Reference image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                ip_adapter_scale = gr.Slider(
                    0.0, 2.0, value=1.0, step=0.05, label="Scale"
                )

        # ── Event wiring (unchanged logic) ───────────────────────────

        def _on_mode_change(value: str):
            is_txt2img = value == handlers.TXT2IMG
            is_img2img = value == handlers.IMG2IMG
            is_inpaint = value in (handlers.INPAINT_MANUAL, handlers.INPAINT_AUTO)
            return (
                gr.update(visible=is_img2img),
                gr.update(visible=is_inpaint),
                gr.update(interactive=is_txt2img),
                gr.update(interactive=is_txt2img),
                gr.update(interactive=not is_txt2img),
                gr.update(interactive=is_inpaint),
            )

        mode.change(
            _on_mode_change,
            inputs=mode,
            outputs=[img2img_image, inpaint_canvas, width, height, strength, mask_blur],
        )

        def _fmt_summary(cfg, steps, w, h, smp):
            return f"CFG **{cfg:.1f}** · Steps **{int(steps)}** · **{int(w)}×{int(h)}** · {smp}"

        for _inp in [cfg_scale, steps, width, height, sampler]:
            _inp.change(
                _fmt_summary,
                inputs=[cfg_scale, steps, width, height, sampler],
                outputs=settings_summary,
            )

        def _refresh_checkpoint_choices(current):
            choices = handlers.checkpoint_choices(ctx)
            values = [value for _, value in choices]
            selected = current if current in values else handlers.default_checkpoint(ctx)
            return gr.update(choices=choices, value=selected)

        refresh_checkpoints.click(
            fn=_refresh_checkpoint_choices,
            inputs=checkpoint,
            outputs=checkpoint,
        )

        def _refresh_all_lora_choices(q):
            return gr.update(choices=handlers.lora_choices(ctx, q))

        def _refresh_after_upload(files, q):
            handlers.upload_lora(ctx, files, q)
            return _refresh_all_lora_choices(q)

        refresh_loras.click(
            fn=_refresh_all_lora_choices,
            inputs=lora_search,
            outputs=lora_table,
        )
        lora_search.change(
            fn=lambda q: gr.update(choices=handlers.lora_choices(ctx, q)),
            inputs=lora_search,
            outputs=lora_table,
        )
        lora_upload.change(
            fn=_refresh_after_upload,
            inputs=[lora_upload, lora_search],
            outputs=lora_table,
        )
        insert_lora_btn.click(
            lambda names, p: handlers.insert_lora_triggers(ctx, names, p),
            inputs=[lora_table, prompt],
            outputs=prompt,
        )

        embedding_search.change(
            fn=lambda q: gr.update(choices=handlers.embedding_choices(ctx, q)),
            inputs=embedding_search,
            outputs=embeddings,
        )
        embedding_upload.change(
            fn=lambda f, q: handlers.upload_embedding(ctx, f, q),
            inputs=[embedding_upload, embedding_search],
            outputs=embeddings,
        )
        insert_embedding_btn.click(
            lambda names, p: handlers.insert_embedding_tokens(ctx, names, p),
            inputs=[embeddings, prompt],
            outputs=prompt,
        )

        controlnet_upload.change(
            fn=lambda f: handlers.upload_controlnet(ctx, f),
            inputs=controlnet_upload,
            outputs=controlnet_model,
        )
        ip_adapter_upload.change(
            fn=lambda f: handlers.upload_ip_adapter(ctx, f),
            inputs=ip_adapter_upload,
            outputs=ip_adapter_name,
        )
        randomize.click(lambda: -1, outputs=seed)

        gen_inputs = [
            mode,
            checkpoint,
            img2img_image,
            canvas_state,
            prompt,
            negative_prompt,
            lora_table,
            embeddings,
            cfg_scale,
            steps,
            sampler,
            seed,
            mask_blur,
            strength,
            width,
            height,
            controlnet_model,
            controlnet_image,
            controlnet_preprocess,
            controlnet_scale,
            controlnet_start,
            controlnet_end,
            ip_adapter_name,
            ip_adapter_image,
            ip_adapter_scale,
            regional_enabled,
            regional_layout,
            regional_ratios,
            regional_base_ratios,
            regional_overlay_ratio,
            regional_use_base_prompt,
            regional_use_common_prompt,
            regional_use_common_negative,
            regional_common_prompt,
            regional_prompt_text,
            regional_negative_text,
            regional_lora_text,
            regional_lora_negative_te,
            regional_lora_negative_unet,
            regional_lora_stop_step,
        ]
        regional_preview_btn.click(
            lambda *args: handlers.preview_regional_masks(ctx, *args),
            inputs=[
                mode,
                img2img_image,
                canvas_state,
                prompt,
                width,
                height,
                regional_enabled,
                regional_layout,
                regional_ratios,
                regional_base_ratios,
                regional_overlay_ratio,
                regional_use_base_prompt,
                regional_use_common_prompt,
                regional_use_common_negative,
                regional_common_prompt,
                regional_prompt_text,
                regional_negative_text,
                regional_lora_text,
                regional_lora_negative_te,
                regional_lora_negative_unet,
                regional_lora_stop_step,
            ],
            outputs=regional_mask_preview,
        )
        gen_event = generate_btn.click(
            lambda *args: handlers.generate_v2(ctx, *args),
            inputs=gen_inputs,
            outputs=[result_image, result_info, used_seed, history_gallery],
        )
        cancel_btn.click(
            fn=lambda: handlers.cancel_generation(ctx),
            inputs=None,
            outputs=None,
            cancels=[gen_event],
        )

        def _send_to_img2img(img):
            if img is None:
                raise gr.Error("No result image")
            return handlers.IMG2IMG, img, gr.update(visible=True), gr.update(visible=False)

        def _send_to_inpaint(img):
            if img is None:
                raise gr.Error("No result image")
            canvas_value = handlers.send_to_inpaint(img)
            return (
                handlers.INPAINT_MANUAL,
                canvas_value,
                canvas_value,
                gr.update(visible=False),
                gr.update(visible=True),
            )

        reedit_img2img.click(
            _send_to_img2img,
            inputs=result_image,
            outputs=[mode, img2img_image, img2img_image, inpaint_canvas],
        )
        reedit_inpaint.click(
            _send_to_inpaint,
            inputs=result_image,
            outputs=[mode, inpaint_canvas, canvas_state, img2img_image, inpaint_canvas],
        )

        preset_save.click(
            lambda *args: handlers.save_preset(ctx, *args),
            inputs=[
                preset_name,
                mode,
                checkpoint,
                prompt,
                negative_prompt,
                lora_table,
                embeddings,
                cfg_scale,
                steps,
                sampler,
                seed,
                mask_blur,
                strength,
                width,
                height,
                controlnet_model,
                controlnet_preprocess,
                controlnet_scale,
                controlnet_start,
                controlnet_end,
                ip_adapter_name,
                ip_adapter_scale,
                regional_enabled,
                regional_layout,
                regional_ratios,
                regional_base_ratios,
                regional_overlay_ratio,
                regional_use_base_prompt,
                regional_use_common_prompt,
                regional_use_common_negative,
                regional_common_prompt,
                regional_prompt_text,
                regional_negative_text,
                regional_lora_text,
                regional_lora_negative_te,
                regional_lora_negative_unet,
                regional_lora_stop_step,
            ],
            outputs=preset_select,
        )

        preset_load.click(
            lambda name: handlers.load_preset(ctx, name),
            inputs=preset_select,
            outputs=[
                mode,
                checkpoint,
                prompt,
                negative_prompt,
                lora_table,
                embeddings,
                cfg_scale,
                steps,
                sampler,
                seed,
                mask_blur,
                strength,
                width,
                height,
                controlnet_model,
                controlnet_preprocess,
                controlnet_scale,
                controlnet_start,
                controlnet_end,
                ip_adapter_name,
                ip_adapter_scale,
                regional_enabled,
                regional_layout,
                regional_ratios,
                regional_base_ratios,
                regional_overlay_ratio,
                regional_use_base_prompt,
                regional_use_common_prompt,
                regional_use_common_negative,
                regional_common_prompt,
                regional_prompt_text,
                regional_negative_text,
                regional_lora_text,
                regional_lora_negative_te,
                regional_lora_negative_unet,
                regional_lora_stop_step,
            ],
        )

    return demo

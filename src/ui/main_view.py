"""Gradio Blocks definition for Img Editor."""
from __future__ import annotations

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


CUSTOM_CSS = """
/* ─── CSS Variables ─────────────────────────────────────── */
:root {
    --ee-bg:           #f4f6fb;
    --ee-surface:      #ffffff;
    --ee-surface-2:    #f8fafc;
    --ee-border:       #d9e0ec;
    --ee-border-strong:#b9c8dc;
    --ee-text:         #1f2937;
    --ee-muted:        #64748b;
    --ee-soft:         #94a3b8;
    --ee-accent:       #7c3aed;
    --ee-accent-soft:  #ede9fe;
    --ee-accent-glow:  rgba(124, 58, 237, 0.14);
    --ee-shadow:       0 1px 3px rgba(15,23,42,0.05), 0 6px 20px rgba(15,23,42,0.07);
}

/* ─── Base ──────────────────────────────────────────────── */
.gradio-container, body, #root, .main {
    background: var(--ee-bg) !important;
    color: var(--ee-text) !important;
    font-family: 'Inter', 'SF Pro Display', system-ui, sans-serif !important;
    min-height: 100vh;
    max-width: none !important;
}
footer { display: none !important; }

/* ─── Scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #eef2f7; }
::-webkit-scrollbar-thumb { background: #bdc8d8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ─── Topbar ─────────────────────────────────────────────── */
#topbar {
    position: sticky !important;
    top: 0 !important;
    z-index: 30 !important;
    background: rgba(255,255,255,0.97) !important;
    border: none !important;
    border-bottom: 1px solid var(--ee-border) !important;
    padding: 10px 20px !important;
    margin-bottom: 0 !important;
    box-shadow: 0 1px 0 rgba(15,23,42,0.05) !important;
}
#topbar .topbar-inner {
    display: flex;
    align-items: center;
    gap: 12px;
    width: 100%;
    min-width: 0;
}
#topbar .topbar-logo {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, #7c3aed, #0f766e);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    font-weight: 800;
    font-size: 14px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(124,58,237,0.22);
}
#topbar .topbar-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--ee-text);
    letter-spacing: 0;
    white-space: nowrap;
}
#topbar .topbar-subtitle {
    font-size: 11px;
    color: #0f766e;
    letter-spacing: 0;
    white-space: nowrap;
}
#topbar .topbar-model {
    margin-left: auto;
    max-width: 42vw;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    font-size: 11px;
    color: var(--ee-muted);
    background: #eef4ff;
    border: 1px solid #d7e4ff;
    border-radius: 999px;
    padding: 3px 10px;
}

/* ─── Diagnostics accordion ──────────────────────────────── */
#diag-accordion {
    border: none !important;
    border-bottom: 1px solid var(--ee-border) !important;
    border-radius: 0 !important;
    background: var(--ee-surface) !important;
    margin: 0 !important;
}
#diag-accordion > .label-wrap {
    padding: 6px 18px !important;
    font-size: 11px !important;
    color: var(--ee-muted) !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    background: var(--ee-surface) !important;
}
#diag-accordion > .label-wrap:hover { color: var(--ee-accent) !important; }

/* ─── Mode selector (Radio → tab bar) ───────────────────── */
#mode-selector {
    background: var(--ee-surface) !important;
    border-bottom: 1px solid var(--ee-border) !important;
    padding: 0 18px !important;
    margin: 0 !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
}
#mode-selector::-webkit-scrollbar { display: none; }
#mode-selector > div,
#mode-selector .wrap {
    flex-direction: row !important;
    gap: 0 !important;
    flex-wrap: nowrap !important;
    min-width: max-content !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
#mode-selector label {
    padding: 10px 22px !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -1px !important;
    color: var(--ee-muted) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    cursor: pointer !important;
    transition: color 0.15s, border-color 0.15s, background 0.15s !important;
    white-space: nowrap !important;
    background: transparent !important;
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}
#mode-selector label:hover {
    color: var(--ee-accent) !important;
    background: var(--ee-accent-soft) !important;
}
#mode-selector label:has(input:checked) {
    color: var(--ee-accent) !important;
    border-bottom-color: var(--ee-accent) !important;
    background: var(--ee-accent-soft) !important;
}
#mode-selector input[type="radio"] { display: none !important; }
#mode-selector span { pointer-events: none; }

/* ─── Main layout ────────────────────────────────────────── */
#main-row {
    gap: 12px !important;
    padding: 12px !important;
    align-items: flex-start !important;
    margin: 0 !important;
}

/* Center panel alignment fix */
#center-panel > .block,
#center-panel > .form {
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* ─── Section labels ─────────────────────────────────────── */
.section-title {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: var(--ee-muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    margin: 0 0 10px 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--ee-border) !important;
    display: block;
}

/* ─── Panels ─────────────────────────────────────────────── */
.panel {
    background: var(--ee-surface) !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    margin-bottom: 10px !important;
    box-shadow: var(--ee-shadow) !important;
}

/* ─── Labels ─────────────────────────────────────────────── */
.gradio-container label > span,
.gradio-container .block > label,
.block label span {
    color: #475569 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

/* ─── Text inputs & Textareas ────────────────────────────── */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: var(--ee-surface) !important;
    border: 1px solid var(--ee-border-strong) !important;
    color: var(--ee-text) !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder {
    color: var(--ee-soft) !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus {
    border-color: var(--ee-accent) !important;
    box-shadow: 0 0 0 3px var(--ee-accent-glow) !important;
    outline: none !important;
}
.gradio-container textarea { resize: vertical !important; }

/* Prompt textarea — main input */
#prompt-box textarea {
    background: var(--ee-surface) !important;
    border-color: #b3a0f5 !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    color: var(--ee-text) !important;
    min-height: 130px !important;
}
#prompt-box textarea:focus {
    border-color: var(--ee-accent) !important;
    box-shadow: 0 0 0 3px var(--ee-accent-glow) !important;
}

/* ─── Dropdowns / Selects ────────────────────────────────── */
.gradio-container select,
.gradio-container .dropdown {
    background: var(--ee-surface) !important;
    border: 1px solid var(--ee-border-strong) !important;
    color: var(--ee-text) !important;
    border-radius: 7px !important;
    min-width: 0 !important;
}
.gradio-container .wrap.svelte-1ipelgc,
.gradio-container .wrap-inner { background: var(--ee-surface) !important; }
.gradio-container .gradio-dropdown,
.gradio-container .gradio-dropdown > div,
.gradio-container .secondary-wrap,
.gradio-container input[role="combobox"],
.gradio-container [role="combobox"],
#sampler-dropdown,
#checkpoint-dropdown {
    background: var(--ee-surface) !important;
    color: var(--ee-text) !important;
    border-color: var(--ee-border-strong) !important;
}
.gradio-container [role="combobox"]:focus-within,
#sampler-dropdown:focus-within,
#checkpoint-dropdown:focus-within {
    border-color: var(--ee-accent) !important;
    box-shadow: 0 0 0 3px var(--ee-accent-glow) !important;
}
.gradio-container [role="listbox"],
.gradio-container .options,
.gradio-container .item,
.gradio-container option {
    background: var(--ee-surface) !important;
    color: var(--ee-text) !important;
}
.gradio-container [role="option"],
.gradio-container .item {
    color: var(--ee-text) !important;
}
.gradio-container [role="option"]:hover,
.gradio-container .item:hover {
    background: var(--ee-accent-soft) !important;
    color: var(--ee-accent) !important;
}

/* ─── Sliders ────────────────────────────────────────────── */
.gradio-container input[type="range"] {
    accent-color: var(--ee-accent) !important;
    cursor: pointer !important;
}
.gradio-container .gradio-slider input[type="number"] {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 12px !important;
    color: var(--ee-muted) !important;
    background: var(--ee-surface-2) !important;
    width: 56px !important;
}

/* ─── Generate button ────────────────────────────────────── */
.primary-run {
    background: var(--ee-accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0 !important;
    height: 48px !important;
    box-shadow: 0 4px 16px rgba(124, 58, 237, 0.24) !important;
    transition: box-shadow 0.2s ease, opacity 0.15s ease, background 0.15s !important;
}
.primary-run:hover {
    background: #6d28d9 !important;
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.36) !important;
    opacity: 1 !important;
}
.primary-run:active { opacity: 0.85 !important; transform: translateY(1px) !important; }
.primary-run:disabled {
    background: #c4b5fd !important;
    color: #ffffff !important;
    box-shadow: none !important;
    animation: btn-pulse 1.8s ease-in-out infinite !important;
}
@keyframes btn-pulse {
    0%, 100% { box-shadow: 0 0 8px rgba(124, 58, 237, 0.15); }
    50%       { box-shadow: 0 0 24px rgba(124, 58, 237, 0.35); }
}

/* ─── Secondary / Cancel button ──────────────────────────── */
.gradio-container button.secondary,
.gradio-container button[variant="secondary"] {
    background: var(--ee-surface) !important;
    border: 1px solid var(--ee-border-strong) !important;
    color: var(--ee-muted) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
.gradio-container button.secondary:hover {
    background: var(--ee-accent-soft) !important;
    border-color: var(--ee-accent) !important;
    color: var(--ee-accent) !important;
}

/* ─── Small utility buttons ──────────────────────────────── */
.gradio-container button.sm,
.gradio-container button[size="sm"] {
    background: var(--ee-surface-2) !important;
    border: 1px solid var(--ee-border) !important;
    color: var(--ee-muted) !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    min-width: 34px !important;
    min-height: 30px !important;
    transition: background 0.15s, color 0.15s !important;
}
.gradio-container button.sm:hover {
    background: var(--ee-accent-soft) !important;
    color: var(--ee-accent) !important;
    border-color: var(--ee-accent) !important;
}

/* ─── "Send to" action buttons ───────────────────────────── */
.send-btn {
    background: transparent !important;
    border: 1px solid var(--ee-border-strong) !important;
    color: var(--ee-muted) !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
.send-btn:hover {
    background: var(--ee-accent-soft) !important;
    border-color: var(--ee-accent) !important;
    color: var(--ee-accent) !important;
}

/* ─── Settings summary bar ───────────────────────────────── */
.settings-summary {
    background: var(--ee-surface-2) !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    margin-bottom: 6px !important;
    text-align: center !important;
}
.settings-summary p {
    font-family: 'SF Mono', 'Cascadia Code', monospace !important;
    font-size: 11px !important;
    color: var(--ee-muted) !important;
    margin: 0 !important;
    letter-spacing: 0 !important;
}

/* ─── Accordion ──────────────────────────────────────────── */
.gradio-container .gradio-accordion {
    background: var(--ee-surface) !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    box-shadow: var(--ee-shadow) !important;
    overflow: hidden !important;
}
.gradio-container .gradio-accordion > .label-wrap {
    padding: 9px 14px !important;
    color: var(--ee-muted) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 0 !important;
    background: var(--ee-surface-2) !important;
    border-bottom: 1px solid var(--ee-border) !important;
}
.gradio-container .gradio-accordion > .label-wrap:hover { color: var(--ee-accent) !important; }

/* ─── LoRA CheckboxGroup ─────────────────────────────────── */
#lora-checkboxes {
    max-height: 280px !important;
    overflow-y: auto !important;
    scrollbar-width: thin !important;
    background: var(--ee-surface-2) !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 8px !important;
}
#lora-checkboxes .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    padding: 4px 4px !important;
}
#lora-checkboxes label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 5px 8px !important;
    border-radius: 5px !important;
    color: var(--ee-text) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
    overflow: hidden !important;
}
#lora-checkboxes label span {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}
#lora-checkboxes label:hover { background: var(--ee-accent-soft) !important; }
#lora-checkboxes input[type="checkbox"] {
    width: 14px !important;
    height: 14px !important;
    accent-color: var(--ee-accent) !important;
    flex-shrink: 0 !important;
    cursor: pointer !important;
}

/* ─── History Gallery ────────────────────────────────────── */
.gradio-container .gradio-gallery { background: transparent !important; }
.gradio-container .gradio-gallery .preview,
.gradio-container .gradio-gallery .thumbnail-item {
    border: 1px solid var(--ee-border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    background: var(--ee-surface) !important;
}
.gradio-container .gradio-gallery .thumbnail-item:hover {
    border-color: var(--ee-accent) !important;
    box-shadow: 0 0 0 2px var(--ee-accent-soft) !important;
}
.gradio-container .gradio-gallery .thumbnail-item.selected {
    border-color: var(--ee-accent) !important;
    box-shadow: 0 0 0 3px var(--ee-accent-glow) !important;
}

/* ─── Image & Canvas components ──────────────────────────── */
.gradio-container .gradio-image {
    border-radius: 8px !important;
    overflow: hidden !important;
    background: var(--ee-surface-2) !important;
    border: 1px solid var(--ee-border) !important;
}
.gradio-container .gradio-imageeditor {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--ee-border) !important;
}

/* ─── Canvas / img2img ───────────────────────────────────── */
#canvas-box,
#img2img-box {
    border: 1px solid var(--ee-border) !important;
    border-radius: 8px !important;
    background: var(--ee-surface-2) !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
}
#canvas-box > div,
#img2img-box > div,
#canvas-box .upload-container,
#canvas-box [data-testid="upload-container"],
#canvas-box .svelte-1ipelgc,
#img2img-box .upload-container,
#img2img-box [data-testid="upload-container"] {
    background: var(--ee-surface-2) !important;
}
#canvas-box p,
#canvas-box .upload-text,
#img2img-box p {
    color: var(--ee-muted) !important;
}
#canvas-box svg,
#img2img-box svg {
    fill: var(--ee-accent) !important;
    color: var(--ee-accent) !important;
}

/* Result image */
#result-image-box {
    background: var(--ee-surface-2) !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    min-height: 280px !important;
}
#result-image-box img {
    max-height: 72vh !important;
    object-fit: contain !important;
    width: 100% !important;
    display: block !important;
}

/* ─── Result meta ────────────────────────────────────────── */
.result-meta,
.result-meta p {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 11px !important;
    color: var(--ee-muted) !important;
    line-height: 1.7 !important;
    padding: 6px 8px !important;
    letter-spacing: 0 !important;
    margin: 4px 0 !important;
    background: var(--ee-surface-2) !important;
    border: 1px solid var(--ee-border) !important;
    border-radius: 7px !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

/* ─── Seed Number input ──────────────────────────────────── */
.seed-number input[type="number"] {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 12px !important;
    color: var(--ee-muted) !important;
    background: var(--ee-surface-2) !important;
}

/* ─── File upload ────────────────────────────────────────── */
.gradio-container .gradio-file {
    background: var(--ee-surface-2) !important;
    border: 1px dashed var(--ee-border-strong) !important;
    border-radius: 7px !important;
    transition: border-color 0.15s !important;
}
.gradio-container .gradio-file:hover {
    border-color: var(--ee-accent) !important;
    background: var(--ee-accent-soft) !important;
}
.gradio-container .gradio-file > div,
.gradio-container .gradio-file .file-preview {
    background: transparent !important;
    color: var(--ee-muted) !important;
}

/* ─── Checkbox group ─────────────────────────────────────── */
.gradio-container .gradio-checkboxgroup label {
    color: var(--ee-text) !important;
    font-size: 12px !important;
}
.gradio-container .gradio-checkboxgroup input[type="checkbox"] {
    accent-color: var(--ee-accent) !important;
}

/* ─── Result image empty state ───────────────────────────── */
#result-image-box .empty,
#result-image-box .icon-wrap {
    background: var(--ee-surface-2) !important;
    color: var(--ee-soft) !important;
}
#result-image-box svg {
    fill: var(--ee-soft) !important;
    color: var(--ee-soft) !important;
}

/* ─── Markdown ───────────────────────────────────────────── */
.gradio-container .gradio-markdown {
    color: var(--ee-muted) !important;
    font-size: 12px !important;
}
.gradio-container .gradio-markdown h3 {
    color: var(--ee-accent) !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    margin: 4px 0 2px !important;
}
.gradio-container .gradio-markdown li { font-size: 12px !important; }
.gradio-container .gradio-markdown code {
    background: var(--ee-accent-soft) !important;
    border-radius: 3px !important;
    padding: 1px 4px !important;
    color: var(--ee-accent) !important;
}

/* ─── Tables ─────────────────────────────────────────────── */
.gradio-container table,
.gradio-container .table-wrap {
    background: var(--ee-surface) !important;
    color: var(--ee-text) !important;
    border-color: var(--ee-border) !important;
}
.gradio-container th {
    background: var(--ee-surface-2) !important;
    color: #334155 !important;
}
.gradio-container td {
    background: var(--ee-surface) !important;
    color: var(--ee-text) !important;
}

@media (max-width: 1100px) {
    #main-row {
        flex-direction: column !important;
    }
    #left-rail,
    #center-panel,
    #right-panel {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
}

@media (max-width: 720px) {
    #topbar { padding: 8px 12px !important; }
    #topbar .topbar-subtitle { display: none; }
    #topbar .topbar-model { max-width: 46vw; }
    #mode-selector { padding: 0 10px !important; }
    #mode-selector label { padding: 9px 14px !important; }
    #main-row { padding: 8px !important; gap: 8px !important; }
    .panel { padding: 10px !important; }
}


"""


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
            diagnostics = gr.Markdown(handlers.diagnostics_markdown(ctx))

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

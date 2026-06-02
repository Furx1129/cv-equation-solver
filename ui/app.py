"""Gradio UI for cv-equation-solver.

Launch: python ui/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import gradio as gr
import numpy as np

from src.expression.display_normalizer import normalize_for_display
from src.expression.normalizer import normalize_tokens
from src.expression.types import ExpressionResult
from src.solver.arithmetic import ArithmeticSolver
from src.solver.symbolic import SymbolicSolver
from src.vision.auto_router import recognize_unknown
from src.vision.calculus_layout import recognize_calculus_layout
from src.vision.deskew import deskew_image
from src.vision.normalization import normalize_formula_image
from src.vision.pipeline import recognize_image
from src.vision.preprocess import PreprocessOptions, preprocess_image, read_image
from src.vision.segmentation import segment_characters
from src.vision.structure_2d import _crop_binary
from src.vision.structure_router import analyze_formula_structure


def _pil_to_cv2(pil_image) -> np.ndarray:
    arr = np.array(pil_image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def _draw_segment_boxes(image: np.ndarray, binary: np.ndarray) -> np.ndarray:
    segments = segment_characters(binary)
    vis = image.copy()
    for seg in segments:
        x, y, w, h = seg.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis


def _draw_token_overlay(image: np.ndarray, tokens) -> np.ndarray:
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for token in tokens:
        x, y, w, h = token.bbox
        if token.text == "UNKNOWN":
            color = (0, 0, 255)
        elif token.confidence < 0.55:
            color = (0, 165, 255)
        else:
            color = (0, 180, 0)

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{token.text} {token.confidence:.2f}"
        text_y = max(14, y - 6)
        cv2.putText(vis, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


def _token_overlay_base_image(image_path: Path, recognition) -> np.ndarray:
    raw = read_image(image_path)
    normalized = normalize_formula_image(raw)
    sources = {token.source for token in recognition.tokens}

    if "printed_template" in sources:
        preprocessed = preprocess_image(
            normalized.image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
        )
        return deskew_image(preprocessed.binary).image

    if "handwritten_rule_template" in sources:
        preprocessed = preprocess_image(
            normalized.image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3, morph_close=2),
        )
        return preprocessed.binary

    if sources & {"structure_2d_template", "calculus_2d_handwritten", "calculus_2d_layout"}:
        preprocessed = preprocess_image(
            normalized.image,
            options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
        )
        return _crop_binary(preprocessed.binary)

    return normalized.image


def _needs_symbolic(expression: str) -> bool:
    if set(expression) <= set("0123456789.+-*/x×÷()= "):
        return False
    symbolic_markers = set("y^∫∫")
    return any(char in symbolic_markers for char in expression) or any(
        name in expression for name in ("sqrt", "integrate", "diff", "limit")
    )


def process(
    image: np.ndarray | None,
    mode: str,
    solver_mode: str,
) -> tuple:
    if image is None:
        return "", "", "", [], None, None, None, "", [], None, "请上传一张算式图片"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "input.png"
        cv2.imwrite(str(tmp_path), image)

        try:
            if mode == "auto":
                decision = recognize_unknown(image_path=tmp_path, solver_timeout_ms=5000)
                recognition = decision.selected.result
                if recognition is None:
                    return "", "", "", [], None, None, None, "", [], None, f"识别失败：{decision.router_reason}"
            else:
                recognition = recognize_image(image_path=tmp_path, backend=mode)
        except Exception as exc:
            return "", "", "", [], None, None, None, "", [], None, f"识别异常：{exc}"

        expression = normalize_tokens(recognition.tokens)
        expression_text = recognition.expression_text

        if solver_mode == "auto":
            if mode == "calculus" or _needs_symbolic(expression_text):
                solver_mode = "symbolic"
            else:
                solver_mode = "arithmetic"

        if solver_mode == "arithmetic":
            solver = ArithmeticSolver()
            solve_result = solver.solve(expression)
        elif solver_mode == "symbolic" and recognition.sympy_text:
            solver = SymbolicSolver()
            solve_result = solver.solve(ExpressionResult(text=recognition.sympy_text, tokens=recognition.tokens, layout=recognition.layout))
        elif solver_mode == "symbolic":
            solver = SymbolicSolver()
            solve_result = solver.solve(expression)
        else:
            solver = ArithmeticSolver()
            solve_result = solver.solve(expression)

        answer = str(solve_result.answer) if solve_result.answer is not None else ""
        error = solve_result.error or ""

        # tokens table: [[idx, text, kind, confidence, bbox]]
        token_rows = []
        for idx, token in enumerate(recognition.tokens):
            token_rows.append([
                idx,
                token.text,
                token.kind or "",
                f"{token.confidence:.3f}",
                f"({token.bbox[0]},{token.bbox[1]},{token.bbox[2]},{token.bbox[3]})",
            ])

        # debug visualizations
        try:
            raw = read_image(tmp_path)
            normalized = normalize_formula_image(raw)
            preprocessed = preprocess_image(
                normalized.image,
                options=PreprocessOptions(threshold_method="fixed", threshold=128, median_kernel=3),
            )
            binary_display = cv2.cvtColor(preprocessed.binary, cv2.COLOR_GRAY2BGR)
            segments_vis = _draw_segment_boxes(normalized.image, preprocessed.binary)
            overlay_base = _token_overlay_base_image(tmp_path, recognition)
            token_overlay = _draw_token_overlay(overlay_base, recognition.tokens)
        except Exception:
            binary_display = None
            segments_vis = None
            token_overlay = None

        # structure info for auto mode
        candidate_rows = []
        if mode == "auto":
            analysis = decision.structure_analysis
            f = analysis.features
            if decision.selected.category == analysis.route_hint:
                route_summary = f"**自动路由**: {decision.selected.category} (置信度 {analysis.route_confidence:.3f})"
            else:
                route_summary = (
                    f"**自动路由**: {analysis.route_hint} -> {decision.selected.category} "
                    f"(置信度 {analysis.route_confidence:.3f})"
                )
            structure_text = (
                f"{route_summary}\n\n"
                f"**主要原因**: {analysis.debug_reason}\n\n"
            )
            for candidate in sorted(decision.candidates, key=lambda item: item.score, reverse=True):
                candidate_rows.append([
                    candidate.role or "",
                    candidate.category,
                    f"{candidate.score:.3f}",
                    "是" if candidate.accepted else "否",
                    candidate.reject_stage or "",
                    candidate.reject_detail or "",
                    candidate.solver_answer or "",
                    candidate.solver_error or "",
                ])
            structure_text += (
                f"**图像特征**:\n"
                f"- 尺寸: {f.width}x{f.height}, 宽高比: {f.aspect_ratio:.2f}\n"
                f"- 连通域: {f.foreground_components}, 长水平线: {f.long_horizontal_lines}\n"
                f"- 分数线: {f.fraction_like_lines}, 堆叠对: {f.stacked_component_pairs}\n"
                f"- 边缘粗糙度: {f.edge_roughness:.3f}, 手写纹理: {f.is_handwritten_texture}\n"
                f"- 积分结构: {f.has_integral_structure}, 网格对齐: {f.is_grid_aligned}\n"
            )
        else:
            structure_text = "（非 auto 模式下不显示结构分析）"

        warnings = "\n".join(recognition.warnings) if recognition.warnings else "无"

        return (
            expression_text,
            answer,
            error,
            token_rows,
            binary_display,
            segments_vis,
            token_overlay,
            structure_text,
            candidate_rows,
            warnings,
            "",
        )


HEADER = """
#  CV Equation Solver — 算式识别与求解

上传算式图片，自动识别表达式并计算结果。支持印刷体四则运算、二维排版、手写符号和微积分。
"""

with gr.Blocks(title="CV Equation Solver") as app:
    gr.Markdown(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.Image(label="上传算式图片", type="numpy", sources=["upload", "clipboard"])
            mode_radio = gr.Radio(
                choices=["printed", "handwritten", "calculus", "auto"],
                value="auto",
                label="识别模式",
                interactive=True,
            )
            solver_radio = gr.Radio(
                choices=["auto", "arithmetic", "symbolic"],
                value="auto",
                label="计算后端",
                interactive=True,
            )
            run_btn = gr.Button("识别并计算", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tab("结果"):
                expression_out = gr.Textbox(label="识别表达式", interactive=False, lines=1)
                with gr.Row():
                    answer_out = gr.Textbox(label="计算结果", interactive=False, scale=2)
                    error_out = gr.Textbox(label="错误", interactive=False, scale=1, container=True)
                tokens_table = gr.Dataframe(
                    headers=["序号", "符号", "类型", "置信度", "BBox"],
                    label="Token 列表",
                    interactive=False,
                )

            with gr.Tab("处理过程"):
                gr.Markdown("### 图像处理")
                with gr.Row():
                    binary_img = gr.Image(label="二值化结果", type="numpy")
                    segments_img = gr.Image(label="字符分割", type="numpy")
                token_overlay_img = gr.Image(label="Token 标注（识别坐标）", type="numpy")
                structure_md = gr.Markdown("### 结构分析\n（运行后显示）")
                candidate_table = gr.Dataframe(
                    headers=["角色", "类别", "分数", "接受", "拒绝阶段", "拒绝原因", "求解结果", "求解错误"],
                    label="Auto 候选分数",
                    interactive=False,
                )
                warnings_out = gr.Textbox(label="警告信息", interactive=False)

    status = gr.Markdown("")

    run_btn.click(
        fn=process,
        inputs=[upload, mode_radio, solver_radio],
        outputs=[
            expression_out, answer_out, error_out, tokens_table,
            binary_img, segments_img, token_overlay_img,
            structure_md, candidate_table, warnings_out, status,
        ],
    )


if __name__ == "__main__":
    launch_kwargs = {"server_name": "127.0.0.1", "share": False}
    if "GRADIO_SERVER_PORT" in os.environ:
        launch_kwargs["server_port"] = int(os.environ["GRADIO_SERVER_PORT"])
    app.launch(**launch_kwargs)

import ast
import os
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")
PROJECT_ROOT = APP_PATH.parents[1]


def _literal_path_from_assignment(name):
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        return _evaluate_path_expr(node.value)
    raise AssertionError(f"{name} assignment not found")


def _evaluate_path_expr(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "join" and len(node.args) >= 2:
            parts = [_evaluate_path_expr(arg) for arg in node.args]
            return os.path.join(*parts)
    if isinstance(node, ast.Name) and node.id == "PROJECT_ROOT":
        return str(PROJECT_ROOT)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise AssertionError(f"Unsupported path expression: {ast.dump(node)}")


def test_tft_model_definition_paths_exist():
    tft_model_path = Path(_literal_path_from_assignment("EXPERIMENT_TFT_MODEL_PATH"))
    multimodal_path = Path(_literal_path_from_assignment("EXPERIMENT_TFT_MULTIMODAL_PATH"))

    assert tft_model_path.exists()
    assert multimodal_path.exists()

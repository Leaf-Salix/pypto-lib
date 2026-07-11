# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source-level checks for Qwen3 prefill runtime-scope placement."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFILL_SOURCE = ROOT / "models" / "qwen3" / "14b" / "prefill_fwd.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _loop(scope: ast.AST, target: str) -> ast.For:
    return next(
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == target
    )


def _direct_loop(scope: ast.AST, target: str) -> ast.For:
    return next(
        node
        for node in scope.body
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == target
    )


def _is_runtime_scope(node: ast.stmt) -> bool:
    if not isinstance(node, ast.With) or len(node.items) != 1:
        return False
    context = node.items[0].context_expr
    return (
        isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and isinstance(context.func.value, ast.Name)
        and context.func.value.id == "pl"
        and context.func.attr == "scope"
    )


def _assert_scope_wrapped(loop: ast.For) -> ast.With:
    assert len(loop.body) == 1
    assert _is_runtime_scope(loop.body[0])
    return loop.body[0]


def _assigned_name(node: ast.stmt) -> str | None:
    if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        return None
    return node.targets[0].id


def _is_create_tensor(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
        return False
    function = node.value.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "pl"
        and function.attr == "create_tensor"
    )


def _target_names(node: ast.Assign) -> set[str]:
    return {
        name.id
        for target in node.targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name)
    }


def _return_names(function: ast.FunctionDef) -> set[str]:
    result = next(node for node in function.body if isinstance(node, ast.Return))
    assert result.value is not None
    return {node.id for node in ast.walk(result.value) if isinstance(node, ast.Name)}


def test_prefill_maps_work_to_all_four_runtime_rings() -> None:
    tree = ast.parse(PREFILL_SOURCE.read_text(encoding="utf-8"))

    entry = _function(tree, "prefill_fwd")
    _assert_scope_wrapped(_loop(entry, "layer_idx"))

    layer = _function(tree, "prefill_layer")
    token_scope = _assert_scope_wrapped(_loop(layer, "p0_idx"))
    rope_scope = next(node for node in token_scope.body if _is_runtime_scope(node))
    final_loop = _direct_loop(token_scope, "final_ti0")

    rope_index = token_scope.body.index(rope_scope)
    parent_allocations = {
        _assigned_name(node) for node in token_scope.body[:rope_index] if _is_create_tensor(node)
    }
    assert {"all_q_padded_tile", "attn_tile"} <= parent_allocations
    assert rope_index < token_scope.body.index(final_loop)
    assert _loop(rope_scope, "rope_core")

    attention_scope = _assert_scope_wrapped(final_loop)
    assert _assigned_name(attention_scope.body[0]) == "finalize_tok"

    attention_calls = {
        node.func.id
        for node in ast.walk(attention_scope)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"_attention_phase_window", "_attention_phase_window_full_single_block"} <= attention_calls


def test_attention_buffer_remains_owned_by_the_parent_scope() -> None:
    tree = ast.parse(PREFILL_SOURCE.read_text(encoding="utf-8"))
    layer = _function(tree, "prefill_layer")

    attn_assignments = [
        node
        for node in ast.walk(layer)
        if isinstance(node, ast.Assign) and "attn_tile" in _target_names(node)
    ]
    assert len(attn_assignments) == 1
    assert _is_create_tensor(attn_assignments[0])

    for helper_name in ("_attention_phase_window", "_attention_phase_window_full_single_block"):
        helper = _function(tree, helper_name)
        assert "attn_tile" not in _return_names(helper)
        assert any(
            isinstance(node, ast.Assign) and "attn_tile" in _target_names(node)
            for node in ast.walk(helper)
        )

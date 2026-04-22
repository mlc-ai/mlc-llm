import pytest
from tvm.target import Target

from mlc_llm.support.auto_target import _apply_webgpu_subgroups

# test category "unittest"
pytestmark = [pytest.mark.unittest]


def test_apply_webgpu_subgroups_enables_webgpu_target():
    target = Target("webgpu")

    updated = _apply_webgpu_subgroups(target, True)

    assert updated is not target
    assert dict(target.export())["supports_subgroups"] is False
    assert dict(updated.export())["supports_subgroups"] is True


def test_apply_webgpu_subgroups_non_webgpu_target_is_unchanged():
    target = Target("llvm")

    updated = _apply_webgpu_subgroups(target, True)

    assert updated is target
    assert dict(updated.export()) == dict(target.export())


@pytest.mark.parametrize("target_kind", ["webgpu", "llvm"])
@pytest.mark.parametrize("enable_subgroups", [False, None])
def test_apply_webgpu_subgroups_disabled_is_unchanged(target_kind, enable_subgroups):
    target = Target(target_kind)

    updated = _apply_webgpu_subgroups(target, enable_subgroups)

    assert updated is target
    assert dict(updated.export()) == dict(target.export())

# pylint: disable=missing-docstring
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from tvm import ir, tir

from .sch_inline import auto_inline_consumers, auto_inline_producers

B, X, Y, K = None, None, None, None


@dataclass
class IterTrait:
    kind: str
    """
    "b" (spatial, x and y)
    "x" (spatial, x-only)
    "y" (spatial, y-only)
    "k" (reduction)
    "s" (spatial, not x or y)
    """
    extent: tir.PrimExpr


def _update_iter_traits_with_buffer_access(
    buffer_access: List[ir.Range],
    traits: List[IterTrait],
    iter2idx: Dict[tir.Var, int],
    x: Optional[str],  # pylint: disable=invalid-name
    y: Optional[str],  # pylint: disable=invalid-name
) -> List[int]:
    trait_kind_map = {}
    if x and y:
        trait_kind_map = {
            "b": "b",
            "s": x,
            x: x,
            y: "b",
            "k": "k",
        }
    buffer_dim_to_iter_idx: List[int] = []
    for dom in buffer_access:
        var = dom.min
        if isinstance(var, tir.Var):
            assert var in iter2idx
            iter_idx = iter2idx[var]
            trait = traits[iter_idx]
            if x and y:
                trait.kind = trait_kind_map[trait.kind]
            buffer_dim_to_iter_idx.append(iter_idx)
        elif isinstance(var, tir.IntImm):
            buffer_dim_to_iter_idx.append(-1)
        else:
            raise ValueError(f"Unknown buffer access in {buffer_access}")
    return buffer_dim_to_iter_idx


def _make_iter_fusion_index_map(
    traits: List[Optional[IterTrait]],
    fuse_to: Dict[str, str],
    result_order: List[str],
) -> tir.IndexMap:
    fused_iters = {
        "b": None,
        "s": None,
        "k": None,
        "x": None,
        "y": None,
    }
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", "int64")
        input_iters.append(v_i)
        if trait is None:
            continue
        kind = fuse_to[trait.kind]
        if fused_iters[kind] is None:
            fused_iters[kind] = v_i
        else:
            fused_iters[kind] = fused_iters[kind] * trait.extent + v_i
    final_indices = [
        fused_iters[c] if fused_iters[c] is not None else tir.const(0, "int64")
        for c in result_order
    ]
    return tir.IndexMap(input_iters, final_indices, None)


def _get_iter_traits(block: tir.Block):
    assert len(block.reads) == 2
    assert len(block.writes) == 1

    traits: List[IterTrait] = [
        IterTrait(
            kind="k" if iter.iter_type == tir.IterVar.CommReduce else "s",
            extent=iter.dom.extent,
        )
        for iter in block.iter_vars
    ]
    iter2idx: Dict[tir.Var, int] = {
        iter_var.var: idx for idx, iter_var in enumerate(block.iter_vars)
    }
    a_to_iter_idx = _update_iter_traits_with_buffer_access(
        block.reads[0].region, traits, iter2idx, "x", "y"
    )
    b_to_iter_idx = _update_iter_traits_with_buffer_access(
        block.reads[1].region, traits, iter2idx, "y", "x"
    )
    c_to_iter_idx = _update_iter_traits_with_buffer_access(
        block.writes[0].region, traits, iter2idx, None, None
    )
    if traits and traits[0].kind in ["x", "y"] and traits[0].extent == 1:
        # HACK: hardcode the 1st iter to 'b'
        traits[0].kind = "b"

    block_index_map = _make_iter_fusion_index_map(
        traits,  # type: ignore[arg-type]
        {
            "b": "b",
            "s": "b",
            "x": "x",
            "y": "y",
            "k": "k",
        },
        ["b", "x", "y", "k"],
    )
    a_index_map = _make_iter_fusion_index_map(
        [traits[i] if i != -1 else None for i in a_to_iter_idx],
        {
            "b": "b",
            "s": "b",
            "x": "s",
            "y": "s",
            "k": "k",
        },
        ["b", "s", "k"],
    )
    b_index_map = _make_iter_fusion_index_map(
        [traits[i] if i != -1 else None for i in b_to_iter_idx],
        {
            "b": "b",
            "s": "b",
            "x": "s",
            "y": "s",
            "k": "k",
        },
        ["b", "s", "k"],
    )
    c_index_map = _make_iter_fusion_index_map(
        [traits[i] if i != -1 else None for i in c_to_iter_idx],
        {
            "b": "b",
            "s": "b",
            "x": "x",
            "y": "y",
            "k": "k",
        },
        ["b", "x", "y"],
    )

    # for i, trait in enumerate(traits):
    #     print(i, trait)
    # print("block_index_map:", block_index_map)
    # print("a_index_map:", a_index_map)
    # print("b_index_map:", b_index_map)
    # print("c_index_map:", c_index_map)

    return block_index_map, a_index_map, b_index_map, c_index_map


def sch_matmul(  # pylint: disable=too-many-statements,too-many-arguments
    name_matmul: str,
    block_size_x=16,
    block_size_y=16,
    vthread_x=2,
    vthread_y=2,
    micro_size_x=4,
    micro_size_y=4,
    micro_size_k=16,
    vector_size=4,
    smem_transpose_a=False,
    smem_transpose_b=False,
) -> Callable[[tir.Schedule], None]:
    def sch_func(
        sch: tir.Schedule,
    ):  # pylint: disable=too-many-branches,too-many-locals
        matmul = sch.get_block(name_matmul)
        (
            block_index_map,
            a_index_map,
            b_index_map,
            c_index_map,
        ) = _get_iter_traits(sch.get(matmul))

        block = sch.cache_read(matmul, 0, "global")
        sch.transform_layout(block, ("write", 0), a_index_map)

        block = sch.cache_read(matmul, 1, "global")
        sch.transform_layout(block, ("write", 0), b_index_map)

        block = sch.cache_write(matmul, 0, "global")
        sch.transform_layout(block, ("read", 0), c_index_map)

        sch.transform_block_layout(matmul, block_index_map)

        global B, X, Y, K  # pylint: disable=global-statement
        B, X, Y, K = [i.dom.extent for i in sch.get(matmul).iter_vars]

        sch.pad_einsum(
            matmul,
            [
                1,
                vthread_x * block_size_x * micro_size_x,
                vthread_y * block_size_y * micro_size_y,
                micro_size_k,
            ],
        )
        # pylint: disable=invalid-name
        _, x, y, k = sch.get_loops(matmul)
        bx, vx, tx, xi = sch.split(x, [None, vthread_x, block_size_x, micro_size_x])
        by, vy, ty, yi = sch.split(y, [None, vthread_y, block_size_y, micro_size_y])
        ko, ki = sch.split(k, factors=[None, micro_size_k])
        # pylint: enable=invalid-name
        sch.reorder(bx, by, vy, vx, ty, tx, ko, ki, yi, xi)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=256)
        sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(matmul, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)

        def _cooperative_fetch(index, vec_len, transpose_shared):
            block = sch.cache_read(matmul, index, "shared")
            num_loops = len(sch.get(block).iter_vars)

            def _make_index_map():
                input_iters = [tir.Var(f"i{j}", "int64") for j in range(num_loops)]
                final_indices = [input_iters[-1]] + input_iters[:-1]
                return tir.IndexMap(input_iters, final_indices, None)

            if transpose_shared:
                sch.transform_layout(block, ("write", 0), index_map=_make_index_map())
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            # pylint: disable=invalid-name
            _, ty, tx, vec = sch.split(
                sch.fuse(*loops),
                factors=[None, block_size_y, block_size_x, vec_len],
            )
            # pylint: enable=invalid-name
            sch.vectorize(vec)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            return block

        def _shared_to_local(index):
            block = sch.cache_read(matmul, index, "local")
            num_loops = len(sch.get(block).iter_vars)
            sch.compute_at(block, ki, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            fused = sch.fuse(*loops)
            sch.vectorize(fused)

        a_g2s = _cooperative_fetch(
            0,
            vec_len=vector_size,
            transpose_shared=smem_transpose_a,
        )
        b_g2s = _cooperative_fetch(
            1,
            vec_len=vector_size,
            transpose_shared=smem_transpose_b,
        )
        _shared_to_local(0)
        _shared_to_local(1)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)
        auto_inline_consumers(sch, l2g)
        sch.decompose_reduction(matmul, ko)

    return sch_func

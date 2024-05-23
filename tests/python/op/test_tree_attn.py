import math
import numpy as np
import pytest
import tvm
import tvm.testing

from mlc_llm.op.tree_attn import tree_attn


@pytest.mark.parametrize("nbatch", [1, 4, 32])
@pytest.mark.parametrize("h_q", [4, 8, 16])
@pytest.mark.parametrize("h_kv", [1, 2, 4])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("rotary_mode", [0])
def test_tree_attn(nbatch, h_q, h_kv, d, rotary_mode):
    np.random.seed(0)
    np.set_printoptions(linewidth=10000)

    def gen_chain(num_nodes):
        mask = np.tril(np.ones((num_nodes, num_nodes)))
        return num_nodes, list(mask.flatten()), np.arange(num_nodes)

    def gen_full_binary_tree(height):
        mask = list()
        pos = list()
        num_nodes = 2**height - 1
        for i in range(num_nodes):
            if i == 0:
                mask_0 = [0] * num_nodes
                mask_0[0] = 1
                mask.append(mask_0)
                pos.append(0)
            else:
                mask_i = mask[(i + 1) // 2 - 1].copy()
                mask_i[i] = 1
                mask.append(mask_i)
                pos.append(pos[(i + 1) // 2 - 1] + 1)
        return num_nodes, list(np.array(mask).flatten()), pos

    ### Inputs
    num_nodes = 0
    m_list = list()
    mn_list = list()
    mask_list = list()
    q_pos_list = list()

    mn_list.append(0)

    for _ in range(nbatch):
        choice = np.random.choice(2, 1, p=[0.5, 0.5])
        if choice == 0:
            nodes_batch = np.random.randint(3, 32)
            res = gen_chain(nodes_batch)
            num_nodes += nodes_batch
        else:
            height = np.random.randint(2, 6)
            res = gen_full_binary_tree(height)
            num_nodes += 2**height - 1
        m_list.append(res[0])
        mn_list.append(res[0] ** 2)
        mask_list.extend(res[1])
        q_pos_list.extend(res[2])

    qkv_indptr = np.array(np.cumsum([0] + m_list)).astype(np.int32)
    m_list = np.array(m_list).astype(np.int32)
    mn_list = np.array(mn_list).astype(np.int32)
    mn_list = np.cumsum(mn_list).astype(np.int32)
    mask_list = np.array(mask_list).astype(np.int32)
    q_pos_list = np.array(q_pos_list).astype(np.int32)

    # print("qkv_indptr:", qkv_indptr)
    # print("m_list:", m_list)
    # print("mn_list:", mn_list)
    # for num_nodes, base in zip(m_list, mn_list):
    #     print("num_nodes:", num_nodes)
    #     print("indptr:", base)
    #     print(
    #         "mask:",
    #         mask_list[base : base + num_nodes * num_nodes].reshape(num_nodes, num_nodes),
    #     )
    #     print("q_pos:", q_pos_list[base : base + num_nodes])

    q = np.random.rand(num_nodes, h_q, d).astype(np.float16)
    q_indptr = qkv_indptr
    k = np.random.rand(num_nodes, h_kv, d).astype(np.float16)
    v = np.random.rand(num_nodes, h_kv, d).astype(np.float16)
    kv_indptr = qkv_indptr
    q_rope_position = q_pos_list
    m_arr = m_list
    mn_indptr = mn_list
    mask = mask_list
    output = np.zeros((num_nodes, h_q, d), dtype=np.float16)
    lse = np.zeros((num_nodes, h_q), dtype=np.float32)
    rotary_scale = 1.0
    rotary_theta = 10000.0
    attn_score_scaling_factor = 1.0

    ### TVM Inputs
    dev = tvm.cuda(0)
    q_tvm = tvm.nd.array(q, dev)
    q_indptr_tvm = tvm.nd.array(q_indptr, dev)
    k_tvm = tvm.nd.array(k, dev)
    v_tvm = tvm.nd.array(v, dev)
    kv_indptr_tvm = tvm.nd.array(kv_indptr, dev)
    q_rope_position_tvm = tvm.nd.array(q_rope_position, dev)
    m_arr_tvm = tvm.nd.array(m_arr, dev)
    mn_indptr_tvm = tvm.nd.array(mn_indptr, dev)
    mask_tvm = tvm.nd.array(mask, dev)
    output_tvm = tvm.nd.array(output, dev)
    lse_tvm = tvm.nd.array(lse, dev)

    target = tvm.target.Target("cuda")
    kernel = tree_attn(h_kv=h_kv, h_q=h_q, d=d, dtype="float16", target=target)
    mod = tvm.build(kernel, target=target)
    mod(
        q_tvm,
        q_indptr_tvm,
        k_tvm,
        v_tvm,
        kv_indptr_tvm,
        q_rope_position_tvm,
        m_arr_tvm,
        mn_indptr_tvm,
        mask_tvm,
        output_tvm,
        lse_tvm,
        rotary_mode,
        rotary_scale,
        rotary_theta,
        attn_score_scaling_factor,
    )

    ### Numpy reference
    def numpy_reference(
        q,
        q_indptr,
        k,
        v,
        kv_indptr,
        q_rope_position,
        m_arr,
        mn_indptr,
        mask,
        output,
        lse,
        rotary_mode,
        rotary_scale,
        rotary_theta,
        attn_score_scaling_factor,
        output_tvm,
    ):
        for i in range(len(m_arr)):
            num_nodes = m_arr[i]
            base = mn_indptr[i]
            q_base = q_indptr[i]
            kv_base = kv_indptr[i]
            q_pos = q_rope_position[base : base + num_nodes]
            q_i = q[q_base : q_base + num_nodes]
            k_i = k[kv_base : kv_base + num_nodes]
            v_i = v[kv_base : kv_base + num_nodes]
            mask_i = mask[base : base + num_nodes * num_nodes].reshape(num_nodes, num_nodes)

            # print(">>>>>>>>>>>>>>>>> i:", i)
            # print("num_nodes:", num_nodes)
            # print("q_pos:", q_pos.shape)
            # print("q_i:", q_i.shape)
            # print("k_i:", k_i.shape)
            # print("v_i:", v_i.shape)
            # print("mask_i:", mask_i)
            # print("mask_i:", mask_i.shape)

            # group attention
            # q: (num_nodes, h_q, d)
            # k: (num_nodes, h_kv, d)
            # v: (num_nodes, h_kv, d)
            group_size = h_q // h_kv
            q_reshape = q_i.transpose(1, 0, 2)  # (h_q, num_nodes, d)
            k_reshape = k_i.transpose(1, 2, 0)  # (h_kv, d, num_nodes)
            v_reshape = v_i.transpose(1, 0, 2)  # (h_kv, num_nodes, d)
            # expand k_reshape
            k_reshape = k_reshape.reshape(h_kv, 1, d, num_nodes)
            k_reshape = np.repeat(k_reshape, group_size, axis=1)
            k_reshape = k_reshape.reshape(h_q, d, num_nodes)
            # expand v_reshape
            v_reshape = v_reshape.reshape(h_kv, 1, num_nodes, d)
            v_reshape = np.repeat(v_reshape, group_size, axis=1)
            v_reshape = v_reshape.reshape(h_q, num_nodes, d)
            # print("q_reshape:", q_reshape.shape)
            # print("k_reshape:", k_reshape.shape)
            # print("v_reshape:", v_reshape.shape)
            # qk: (h_q, num_nodes, num_nodes)
            assert rotary_mode == 0
            qk = np.matmul(q_reshape, k_reshape) * attn_score_scaling_factor / math.sqrt(float(d))
            # softmax(qk, axis=-1), numerical stability
            qk[:, mask_i == 0] = -np.inf
            qk_max = np.max(qk, axis=-1, keepdims=True)
            qk = np.exp(qk - qk_max)
            qk = qk / np.sum(qk, axis=-1, keepdims=True)
            # print(qk)
            # attention
            output_i = np.matmul(qk, v_reshape).transpose(1, 0, 2)  # (num_nodes, h_q, d)
            # print(output_i)

            tvm.testing.assert_allclose(
                output_i, output_tvm[q_base : q_base + num_nodes], rtol=1e-3, atol=1e-3
            )

    numpy_reference(
        q,
        q_indptr,
        k,
        v,
        kv_indptr,
        q_rope_position,
        m_arr,
        mn_indptr,
        mask,
        output,
        lse,
        rotary_mode,
        rotary_scale,
        rotary_theta,
        attn_score_scaling_factor,
        output_tvm.asnumpy(),
    )


if __name__ == "__main__":
    tvm.testing.main()

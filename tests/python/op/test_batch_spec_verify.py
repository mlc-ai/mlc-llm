import numpy as np
import pytest
import tvm
import tvm.testing

from mlc_llm.op.batch_spec_verify import batch_spec_verify

# test category "op_correctness"
pytestmark = [pytest.mark.op_correctness]


@pytest.mark.parametrize("nbatch", [32, 64])
@pytest.mark.parametrize("vocab", [3, 32, 64, 32000, 33, 65, 32001, 128000])
@pytest.mark.parametrize("plist", [[0.5, 0.5], [1, 0], [0, 1]])
def test_batch_spec_verify(nbatch, vocab, plist):
    def numpy_reference(
        draft_probs,
        draft_tokens,
        model_probs,
        token_tree_first_child,
        token_tree_next_sibling,
        uniform_samples,
        token_tree_parent_ptr,
    ):
        nbatch = token_tree_parent_ptr.shape[0]
        for b in range(nbatch):
            parent_ptr = token_tree_parent_ptr[b]
            child_ptr = token_tree_first_child[parent_ptr]
            while child_ptr != -1:
                child_token = draft_tokens[child_ptr]
                p_child = model_probs[parent_ptr, child_token]
                q_child = draft_probs[child_ptr, child_token]
                uniform_sample = uniform_samples[child_ptr]
                if p_child / q_child >= uniform_sample:
                    parent_ptr = child_ptr
                    child_ptr = token_tree_first_child[child_ptr]
                else:
                    model_probs[parent_ptr, :] = np.maximum(
                        model_probs[parent_ptr, :] - draft_probs[child_ptr, :], 0.0
                    )
                    psum = np.sum(model_probs[parent_ptr, :])
                    model_probs[parent_ptr, :] /= psum
                    child_ptr = token_tree_next_sibling[child_ptr]
            token_tree_parent_ptr[b] = parent_ptr

    np.random.seed(0)

    def gen_chain(num_nodes, base):
        token_tree_first_child = list()
        token_tree_next_sibling = list()
        for i in range(num_nodes):
            token_tree_first_child.append(base + i + 1 if i + 1 < num_nodes else -1)
            token_tree_next_sibling.append(-1)
        return token_tree_first_child, token_tree_next_sibling, base, base + 1

    def gen_full_binary_tree(height, base):
        token_tree_first_child = list()
        token_tree_next_sibling = list()
        num_nodes = 2**height - 1
        for i in range(num_nodes):
            token_tree_first_child.append(base + i * 2 + 1 if i * 2 + 1 < num_nodes else -1)
            token_tree_next_sibling.append(base + i * 2 + 2 if i * 2 + 2 < num_nodes else -1)
        return token_tree_first_child, token_tree_next_sibling, base, base + 1

    ### Inputs
    num_nodes = 0
    token_tree_first_child = list()
    token_tree_next_sibling = list()
    token_tree_parent_ptr = list()

    for _ in range(nbatch):
        choice = np.random.choice(2, 1, p=plist)
        if choice == 0:
            nodes_batch = np.random.randint(3, 32)
            res = gen_chain(nodes_batch, num_nodes)
            num_nodes += nodes_batch
        else:
            height = np.random.randint(3, 5)
            res = gen_full_binary_tree(height, num_nodes)
            num_nodes += 2**height - 1
        token_tree_first_child.extend(res[0])
        token_tree_next_sibling.extend(res[1])
        token_tree_parent_ptr.append(res[2])

    token_tree_first_child = np.array(token_tree_first_child).astype("int32")
    token_tree_next_sibling = np.array(token_tree_next_sibling).astype("int32")
    token_tree_parent_ptr = np.array(token_tree_parent_ptr).astype("int32")

    draft_probs = np.random.rand(num_nodes, vocab).astype("float32")
    draft_probs /= np.sum(draft_probs, axis=1, keepdims=True)
    draft_tokens = np.random.randint(0, vocab, num_nodes).astype("int32")
    model_probs = np.random.rand(num_nodes, vocab).astype("float32")
    model_probs /= np.sum(model_probs, axis=1, keepdims=True)
    uniform_samples = np.random.rand(num_nodes).astype("float32")

    ### TVM Inputs
    dev = tvm.cuda(0)
    draft_probs_tvm = tvm.nd.array(draft_probs, dev)
    draft_tokens_tvm = tvm.nd.array(draft_tokens, dev)
    model_probs_tvm = tvm.nd.array(model_probs, dev)
    token_tree_first_child_tvm = tvm.nd.array(token_tree_first_child, dev)
    token_tree_next_sibling_tvm = tvm.nd.array(token_tree_next_sibling, dev)
    uniform_samples_tvm = tvm.nd.array(uniform_samples, dev)
    token_tree_parent_ptr_tvm = tvm.nd.array(token_tree_parent_ptr, dev)

    # print("draft_probs", draft_probs)
    # print("draft_tokens", draft_tokens)
    # print("model_probs", model_probs)
    # print("token_tree_first_child", token_tree_first_child)
    # print("token_tree_next_sibling", token_tree_next_sibling)
    # print("uniform_samples", uniform_samples)
    # print("token_tree_parent_ptr", token_tree_parent_ptr)

    ### Numpy reference
    numpy_reference(
        draft_probs,
        draft_tokens,
        model_probs,
        token_tree_first_child,
        token_tree_next_sibling,
        uniform_samples,
        token_tree_parent_ptr,
    )
    # print("model_probs", model_probs)
    # print("token_tree_parent_ptr", token_tree_parent_ptr)

    ### TVM
    kernel = batch_spec_verify(vocab)
    mod = tvm.build(kernel, target="cuda")
    mod(
        draft_probs_tvm,
        draft_tokens_tvm,
        model_probs_tvm,
        token_tree_first_child_tvm,
        token_tree_next_sibling_tvm,
        uniform_samples_tvm,
        token_tree_parent_ptr_tvm,
    )
    # print("model_probs", model_probs_tvm.asnumpy())
    # print("token_tree_parent_ptr", token_tree_parent_ptr_tvm.asnumpy())

    tvm.testing.assert_allclose(model_probs, model_probs_tvm.asnumpy())
    tvm.testing.assert_allclose(
        token_tree_parent_ptr, token_tree_parent_ptr_tvm.asnumpy(), rtol=0, atol=0
    )

    time_evaluator = mod.time_evaluator(mod.entry_name, dev, number=10, repeat=3)
    print(f"batch_size: {nbatch}, vocab_size: {vocab}, tree_structure: {plist}")
    print(
        time_evaluator(
            draft_probs_tvm,
            draft_tokens_tvm,
            model_probs_tvm,
            token_tree_first_child_tvm,
            token_tree_next_sibling_tvm,
            uniform_samples_tvm,
            token_tree_parent_ptr_tvm,
        )
    )


if __name__ == "__main__":
    tvm.testing.main()

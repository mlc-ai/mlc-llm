import numpy as np
import pytest
import tvm
import tvm.testing

from mlc_llm.op.batch_spec_verify import batch_spec_verify


@pytest.mark.parametrize("vocab", [32, 64, 32000, 33, 65, 32001])
def test_batch_spec_verify(vocab):
    def numpy_reference(
        draft_probs,
        draft_tokens,
        model_probs,
        token_tree_first_child,
        token_tree_next_sibling,
        uniform_samples,
        token_tree_parent_ptr,
        token_tree_child_ptr,
    ):
        nbatch = token_tree_parent_ptr.shape[0]
        for b in range(nbatch):
            parent_ptr = token_tree_parent_ptr[b]
            child_ptr = token_tree_child_ptr[b]
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
            token_tree_child_ptr[b] = child_ptr

    np.random.seed(0)
    nbatch = 32
    height = 5
    num_nodes_per_batch = 2**height - 1
    num_nodes = num_nodes_per_batch * nbatch

    ### Inputs
    draft_probs = np.random.rand(num_nodes, vocab).astype("float32")
    draft_probs /= np.sum(draft_probs, axis=1, keepdims=True)
    draft_tokens = np.random.randint(0, vocab, num_nodes).astype("int32")
    model_probs = np.random.rand(num_nodes, vocab).astype("float32")
    model_probs /= np.sum(model_probs, axis=1, keepdims=True)
    # binary tree
    token_tree_first_child = list()
    token_tree_next_sibling = list()
    for b in range(nbatch):
        for i in range(num_nodes_per_batch):
            token_tree_first_child.append(
                b * num_nodes_per_batch + i * 2 + 1 if i * 2 + 1 < num_nodes_per_batch else -1
            )
            token_tree_next_sibling.append(
                b * num_nodes_per_batch + i * 2 + 2 if i * 2 + 2 < num_nodes_per_batch else -1
            )
    token_tree_first_child = np.array(token_tree_first_child).astype("int32")
    token_tree_next_sibling = np.array(token_tree_next_sibling).astype("int32")
    uniform_samples = np.random.rand(num_nodes).astype("float32")
    token_tree_parent_ptr = list()
    token_tree_child_ptr = list()
    for b in range(nbatch):
        token_tree_parent_ptr.append(b * num_nodes_per_batch)
        token_tree_child_ptr.append(b * num_nodes_per_batch + 1)
    token_tree_parent_ptr = np.array(token_tree_parent_ptr).astype("int32")
    token_tree_child_ptr = np.array(token_tree_child_ptr).astype("int32")

    ### TVM Inputs
    dev = tvm.cuda(0)
    draft_probs_tvm = tvm.nd.array(draft_probs, dev)
    draft_tokens_tvm = tvm.nd.array(draft_tokens, dev)
    model_probs_tvm = tvm.nd.array(model_probs, dev)
    token_tree_first_child_tvm = tvm.nd.array(token_tree_first_child, dev)
    token_tree_next_sibling_tvm = tvm.nd.array(token_tree_next_sibling, dev)
    uniform_samples_tvm = tvm.nd.array(uniform_samples, dev)
    token_tree_parent_ptr_tvm = tvm.nd.array(token_tree_parent_ptr, dev)
    token_tree_child_ptr_tvm = tvm.nd.array(token_tree_child_ptr, dev)

    # print("draft_probs", draft_probs)
    # print("draft_tokens", draft_tokens)
    # print("model_probs", model_probs)
    # print("token_tree_first_child", token_tree_first_child)
    # print("token_tree_next_sibling", token_tree_next_sibling)
    # print("uniform_samples", uniform_samples)
    # print("token_tree_parent_ptr", token_tree_parent_ptr)
    # print("token_tree_child_ptr", token_tree_child_ptr)

    ### Numpy reference
    numpy_reference(
        draft_probs,
        draft_tokens,
        model_probs,
        token_tree_first_child,
        token_tree_next_sibling,
        uniform_samples,
        token_tree_parent_ptr,
        token_tree_child_ptr,
    )
    # print("model_probs", model_probs)
    # print("token_tree_parent_ptr", token_tree_parent_ptr)
    # print("token_tree_child_ptr", token_tree_child_ptr)

    ### TVM
    kernel = batch_spec_verify()
    mod = tvm.build(kernel, target="cuda")
    mod(
        draft_probs_tvm,
        draft_tokens_tvm,
        model_probs_tvm,
        token_tree_first_child_tvm,
        token_tree_next_sibling_tvm,
        uniform_samples_tvm,
        token_tree_parent_ptr_tvm,
        token_tree_child_ptr_tvm,
    )
    # print("model_probs", model_probs_tvm.asnumpy())
    # print("token_tree_parent_ptr", token_tree_parent_ptr_tvm.asnumpy())
    # print("token_tree_child_ptr", token_tree_child_ptr_tvm.asnumpy())

    tvm.testing.assert_allclose(model_probs, model_probs_tvm.asnumpy())
    print(token_tree_parent_ptr)
    print(token_tree_parent_ptr_tvm.asnumpy())
    tvm.testing.assert_allclose(token_tree_parent_ptr, token_tree_parent_ptr_tvm.asnumpy(), rtol=0, atol=0)
    tvm.testing.assert_allclose(token_tree_child_ptr, token_tree_child_ptr_tvm.asnumpy(), rtol=0, atol=0)


if __name__ == "__main__":
    tvm.testing.main()

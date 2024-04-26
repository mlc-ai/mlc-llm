"""Operators for batch verify in speculative decoding."""

from tvm.script import tir as T

# mypy: disable-error-code="attr-defined,valid-type,name-defined"
# pylint: disable=too-many-locals,invalid-name,too-many-arguments,
# pylint: disable=too-many-statements,line-too-long,too-many-nested-blocks,too-many-branches


def batch_spec_verify(vocab_size):
    """Batch draft verify function. This function verifies the token tree.

    Before calling the function

    - token_tree_parent_ptr[b] should store the root of the tree

    - draft_probs[node_id, :] stores the prob that samples the correspond tree node
    - model_probs[node_id, :] stores the prob that should be used to sample its children
    - Please note that the storage convention difference between model_probs and draft_probs
        draft_probs was stored on the token node, while model_probs stores on the parent.
        This is an intentional design since we can sample different child token with different
        proposal draft probabilities, but the ground truth model_prob is unique per parent.

    After calling the function
    - token_tree_parent_ptr[b] points to the last token accepted
    - There should be a followup sample step that samples from model_probs[token_tree_parent_ptr[b], :]
        This token will be appended to the token generated.

    This function will inplace update model_probs if a token was rejected and renormalization is needed.

    Parameters
    ----------
    draft_probs:
        The draft probability attached to each tree node

    draft_tokens:
        The draft token in each node

    model_probs:
        The model proability attached to each parent

    token_tree_first_child:
        The first child of each tree node, if there is no child, it should be -1

    token_tree_next_sibling
        The next sibling of each tree node, if there is no next sibling, it should be -1

    uniform_samples
        Per node uniform sample used to check rejection

    token_tree_parent_ptr:
        Current parent ptr state
    """
    TX = 1024

    def _var(dtype="int32"):
        return T.alloc_buffer((1,), dtype, scope="local")

    # fmt: off
    @T.prim_func(private=True)
    def _func(
        var_draft_probs: T.handle,
        var_draft_tokens: T.handle,
        var_model_probs: T.handle,
        var_token_tree_first_child: T.handle,
        var_token_tree_next_sibling: T.handle,
        var_uniform_samples: T.handle,
        var_token_tree_parent_ptr: T.handle,
    ):
        """
        [
            blockIdx.x on batch,
            threadIdx.x on vocab_size,
            for loop over excessive amounts
        ]
        """
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
        num_nodes = T.int32(is_size_var=True)
        nbatch = T.int32(is_size_var=True)

        draft_probs = T.match_buffer(var_draft_probs, (num_nodes, vocab_size), "float32")
        draft_tokens = T.match_buffer(var_draft_tokens, (num_nodes,), "int32")
        model_probs = T.match_buffer(var_model_probs, (num_nodes, vocab_size), "float32")
        token_tree_first_child = T.match_buffer(var_token_tree_first_child, (num_nodes,), "int32")
        token_tree_next_sibling = T.match_buffer(var_token_tree_next_sibling, (num_nodes,), "int32")
        uniform_samples = T.match_buffer(var_uniform_samples, (num_nodes,), "float32")
        token_tree_parent_ptr = T.match_buffer(var_token_tree_parent_ptr, (nbatch,), "int32")

        with T.block("kernel"):
            child_ptr = _var()
            parent_ptr = _var()
            child_token = _var()
            done = _var("bool")
            psum = _var("float32")
            t0 = _var("float32")
            model_prob_local = _var("float32")
            draft_prob_local = _var("float32")
            p_child = _var("float32")
            q_child = _var("float32")
            uniform_sample = _var("float32")

            pred_shared = T.alloc_buffer((1,), "bool", scope="shared")
            pred_local = T.alloc_buffer((1,), "bool", scope="local")

            for _bx in T.thread_binding(0, nbatch, thread="blockIdx.x"):
                for _tx in T.thread_binding(0, TX, thread="threadIdx.x"):
                    with T.block("CTA"):
                        # batch size
                        b = T.axis.S(nbatch, _bx)
                        tx = T.axis.S(TX, _tx)

                        parent_ptr[0] = token_tree_parent_ptr[b]
                        child_ptr[0] = token_tree_first_child[parent_ptr[0]]
                        done[0] = False

                        while T.Not(done[0]):
                            T.tvm_storage_sync("shared") # ensure all effects last round are visible
                            if child_ptr[0] == -1:
                                done[0] = True
                                T.tvm_storage_sync("shared") # sync before exit
                            else:
                                # decide to validate current ptr
                                if tx == 0:
                                    child_token[0] = draft_tokens[child_ptr[0]]
                                    p_child[0] = model_probs[parent_ptr[0], child_token[0]]
                                    q_child[0] = draft_probs[child_ptr[0], child_token[0]]
                                    uniform_sample[0] = uniform_samples[child_ptr[0]]
                                    pred_shared[0] = p_child[0] >= uniform_sample[0] * q_child[0]  # use multiplication to avoid division by zero
                                T.tvm_storage_sync("shared") # make sure all read of model_probs are done
                                pred_local[0] = pred_shared[0]

                                # accept the proposal, we move to child
                                if pred_local[0]:
                                    parent_ptr[0] = child_ptr[0]
                                    child_ptr[0] = token_tree_first_child[child_ptr[0]]
                                else:
                                    psum[0] = 0.0
                                    # renormalize probability, predicated by stopped_expansion[b]:
                                    for i in T.serial(T.ceildiv(vocab_size, TX)):
                                        k = T.meta_var(i * TX + tx)
                                        if k < vocab_size:
                                            model_prob_local[0] = model_probs[parent_ptr[0], k]
                                            draft_prob_local[0] = draft_probs[child_ptr[0], k]
                                            model_prob_local[0] = T.max(model_prob_local[0] - draft_prob_local[0], 0.0)
                                            psum[0] += model_prob_local[0]

                                    with T.block("block_cross_thread"):
                                        T.reads(psum[0])
                                        T.writes(t0[0])
                                        T.attr(
                                            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                            "reduce_scope",
                                            T.reinterpret("handle", T.uint64(0)),
                                        )
                                        T.tvm_thread_allreduce(T.uint32(1), psum[0], True, t0[0], tx, dtype="handle")

                                    if t0[0] < 1e-7:
                                        # accept the proposal, we move to child
                                        parent_ptr[0] = child_ptr[0]
                                        child_ptr[0] = token_tree_first_child[child_ptr[0]]
                                    else:
                                        # renormalize
                                        for i in T.serial(T.ceildiv(vocab_size, TX)):
                                            k = T.meta_var(i * TX + tx)
                                            if k < vocab_size:
                                                model_prob_local[0] = model_probs[parent_ptr[0], k]
                                                draft_prob_local[0] = draft_probs[child_ptr[0], k]
                                                model_prob_local[0] = T.max(model_prob_local[0] - draft_prob_local[0], 0.0)
                                                model_probs[parent_ptr[0], k] = model_prob_local[0] / t0[0]

                                        child_ptr[0] = token_tree_next_sibling[child_ptr[0]]

                        if tx == 0:
                            token_tree_parent_ptr[b] = parent_ptr[0]
    # fmt: on

    return _func

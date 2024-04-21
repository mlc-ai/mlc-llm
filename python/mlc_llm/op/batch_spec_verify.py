"""Operators for batch verify in speculative decoding."""

from tvm.script import tir as T

# mypy: disable-error-code="attr-defined,valid-type,name-defined"
# pylint: disable=too-many-locals,invalid-name,too-many-arguments,too-many-statements,line-too-long,too-many-nested-blocks


def batch_spec_verify():
    """Return the TIR function for batch verify in speculative decoding."""
    TX = 128
    VEC = 4

    def _var(dtype="int32"):
        return T.alloc_buffer((1,), dtype, scope="local")

    # fmt: off
    @T.prim_func(check_well_formed=False, private=True)
    def _func(
        var_draft_probs: T.handle,
        var_draft_tokens: T.handle,
        var_model_probs: T.handle,
        var_token_tree_first_child: T.handle,
        var_token_tree_next_sibling: T.handle,
        var_uniform_samples: T.handle,
        var_token_tree_parent_ptr: T.handle,
        var_token_tree_child_ptr: T.handle,
    ):
        """
        [
            blockIdx.x on batch,
            threadIdx.x on vocab,
            for loop over excessive amounts
        ]
        """
        T.func_attr({"tir.is_scheduled": 1, "tir.noalias": True})
        num_nodes = T.int32(is_size_var=True)
        vocab = T.int32(is_size_var=True)
        nbatch = T.int32(is_size_var=True)

        draft_probs = T.match_buffer(var_draft_probs, (num_nodes, vocab), "float32")
        draft_tokens = T.match_buffer(var_draft_tokens, (num_nodes,), "int32")
        model_probs = T.match_buffer(var_model_probs, (num_nodes, vocab), "float32")
        token_tree_first_child = T.match_buffer(var_token_tree_first_child, (num_nodes,), "int32")
        token_tree_next_sibling = T.match_buffer(var_token_tree_next_sibling, (num_nodes,), "int32")
        uniform_samples = T.match_buffer(var_uniform_samples, (num_nodes,), "float32")
        token_tree_parent_ptr = T.match_buffer(var_token_tree_parent_ptr, (nbatch,), "int32")
        token_tree_child_ptr = T.match_buffer(var_token_tree_child_ptr, (nbatch,), "int32")

        for _bx in T.thread_binding(0, nbatch, thread="blockIdx.x"):
            for _tx in T.thread_binding(0, TX, thread="threadIdx.x"):
                with T.block("CTA"):
                    child_ptr = _var()
                    parent_ptr = _var()
                    child_token = _var()
                    done = _var("bool")
                    psum = _var("float32")
                    t0 = _var("float32")
                    model_prob_local = T.alloc_buffer((VEC,), "float32", scope="local")
                    draft_prob_local = T.alloc_buffer((VEC,), "float32", scope="local")

                    # batch size
                    b = T.axis.S(nbatch, _bx)
                    tx = T.axis.S(TX, _tx)
                    # or simmply while true
                    parent_ptr[0] = token_tree_parent_ptr[b]
                    child_ptr[0] = token_tree_child_ptr[b]
                    done[0] = False

                    while T.Not(done[0]):
                        if child_ptr[0] == -1:
                            done[0] = True
                        else:
                            # decide to validate current ptr
                            child_token[0] = draft_tokens[child_ptr[0]]
                            p_child = T.meta_var(model_probs[parent_ptr[0], child_token[0]])
                            q_child = T.meta_var(draft_probs[child_ptr[0], child_token[0]])
                            uniform_sample = T.meta_var(uniform_samples[child_ptr[0]])
                            # accept the proposal, we move to child
                            if p_child / q_child >= uniform_sample:
                                parent_ptr[0] = child_ptr[0]
                                child_ptr[0] = token_tree_first_child[child_ptr[0]]
                            else:
                                psum[0] = 0.0
                                # renormalize probability, predicated by stopped_expansion[b]:
                                for i in T.serial(T.ceildiv(vocab, TX * VEC)):
                                    for vec in T.vectorized(VEC):
                                        k = T.meta_var(i * TX * VEC + tx * VEC + vec)
                                        model_prob_local[vec] = T.if_then_else(k < vocab, model_probs[parent_ptr[0], k], 0.0)
                                    for vec in T.vectorized(VEC):
                                        k = T.meta_var(i * TX * VEC + tx * VEC + vec)
                                        draft_prob_local[vec] = T.if_then_else(k < vocab, draft_probs[child_ptr[0], k], 0.0)
                                    for vec in T.serial(VEC): # vectorize?
                                        model_prob_local[vec] = T.max(model_prob_local[vec] - draft_prob_local[vec], 0.0)
                                    for vec in T.vectorized(VEC):
                                        k = T.meta_var(i * TX * VEC + tx * VEC + vec)
                                        if k < vocab:
                                            model_probs[parent_ptr[0], k] = model_prob_local[vec]
                                    for vec in T.serial(VEC):
                                        psum[0] += model_prob_local[vec]

                                with T.block("block_cross_thread"):
                                    T.reads(psum[0])
                                    T.writes(t0[0])
                                    T.attr(
                                        T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                        "reduce_scope",
                                        T.reinterpret("handle", T.uint64(0)),
                                    )
                                    T.tvm_thread_allreduce(T.uint32(1), psum[0], True, t0[0], tx, dtype="handle")

                                # renormalize
                                for i in T.serial(T.ceildiv(vocab, TX * VEC)):
                                    for vec in T.vectorized(VEC):
                                        k = T.meta_var(i * TX * VEC + tx * VEC + vec)
                                        if k < vocab:
                                            model_probs[parent_ptr[0], k] = model_probs[parent_ptr[0], k] / t0[0]

                                child_ptr[0] = token_tree_next_sibling[child_ptr[0]]

                    token_tree_parent_ptr[b] = parent_ptr[0]
                    token_tree_child_ptr[b] = child_ptr[0]
    # fmt: on

    return _func

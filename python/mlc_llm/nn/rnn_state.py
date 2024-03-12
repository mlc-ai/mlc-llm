"""RNN State modeling."""

from typing import Sequence, Union

from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor
from tvm.script import tir as T


class RNNState(Object):
    """The RNN State used in Space State Models"""

    @staticmethod
    def create(
        max_batch_size: tir.Var,
        num_hidden_layers: int,
        max_history: int,
        init_values: Sequence[Tensor],
        name: str = "rnn_state",
    ) -> "RNNState":
        """Create a RNN state object.

        Parameters
        ----------
        max_batch_size : tir.Var
            The maximum batch size.
        num_hidden_layers : int
            The number of hidden layers.
        max_history : int
            The maximum history length.
        init_values : Sequence[Tensor]
            The initial values of the RNN state.
        """

        bb = rx.BlockBuilder.current()
        state_infos = [(v.shape, v.dtype) for v in init_values]

        f_gets = [
            bb.add_func(
                RNNState.create_get_func(shape, dtype, max_batch_size, max_history, id),
                f"rnn_state_get_{id}",
            )
            for id, (shape, dtype) in enumerate(state_infos)
        ]
        f_sets = [
            bb.add_func(
                RNNState.create_set_func(shape, dtype, max_batch_size, max_history, id),
                f"rnn_state_set_{id}",
            )
            for id, (shape, dtype) in enumerate(state_infos)
        ]

        ret = RNNState(
            _expr=rx.call_pure_packed(
                "vm.builtin.rnn_state_create",
                rx.PrimValue(num_hidden_layers),
                max_batch_size,
                max_history,
                f_gets,
                f_sets,
                [v._expr for v in init_values],  # pylint: disable=protected-access
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            _name=name,
        )
        return ret

    def get(
        self,
        layer_id: int,
        state_id: int,
        shape: Sequence[tir.PrimExpr],
        dtype: str,
    ) -> Tensor:
        """Get the state of the RNN layer.

        - If there is only one sequence, we can directly use the storage memory,
        without copying the data.
        - If there are multiple sequences, we need to copy the data to get a contiguous
        memory.

        Parameters
        ----------
        layer_id : int
            The layer id.
        state_id : int
            The state id.
        shape : Sequence[tir.PrimExpr]
            The shape of the state tensor.
        dtype: str
            The data type of the state tensor.

        Returns
        -------
        Tensor
            The state tensor, with shape `(batch_size, *state_size)`.
        """
        bb = rx.BlockBuilder.current()

        return Tensor(
            _expr=bb.emit(
                rx.call_dps_packed(
                    "vm.builtin.rnn_state_get",
                    [self._expr, layer_id, state_id],
                    out_sinfo=rx.TensorStructInfo(shape, dtype),
                )
            )
        )

    def set(self, layer_id: int, state_id: int, value: Tensor) -> "RNNState":
        """Set the state of the RNN layer.

        Parameters
        ----------
        layer_id : int
            The layer id.
        state_id : int
            The state id.
        value : Tensor
            The state tensor, with shape `(batch_size, *state_size)`.
        """
        bb = rx.BlockBuilder.current()
        return RNNState(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.rnn_state_set",
                    self._expr,
                    rx.PrimValue(layer_id),
                    rx.PrimValue(state_id),
                    value._expr,  # pylint: disable=protected-access
                    sinfo_args=[rx.ObjectStructInfo()],
                )
            ),
            _name="rnn_state_set",
        )

    @staticmethod
    def create_get_func(
        shape: Sequence[Union[int, tir.Var]],
        dtype: str,
        max_batch_size: Union[int, tir.Var],
        max_history: Union[int, tir.Var],
        state_id: int,
    ) -> tir.PrimFunc:
        """Create the get function with given state shape.

        Parameters
        ----------
        shape : Sequence[Union[int, tir.Var]]
            The shape of the state tensor.

        dtype: str
            The data type of the state tensor.

        max_batch_size : Union[int, tir.Var]
            The maximum batch size.

        max_history : Union[int, tir.Var]
            The maximum history length.

        state_id : int
            The id of the state, used for naming the function.

        Returns
        -------
        tir.PrimFunc
            The get function.
        """

        def _func_one_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_output: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_get_{state_id}"})

                storage = T.match_buffer(
                    var_storage, (max_batch_size, max_history, shape[0]), dtype
                )
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                output = T.match_buffer(var_output, (batch_size, shape[0]), dtype)

                for i in range(batch_size):
                    for s in range(shape[0]):
                        with T.block("copy"):
                            vi, vs = T.axis.remap("SS", [i, s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = history_slot_ids[vi]
                            output[vi, vs] = storage[seq_id, history_id, vs]

            return f

        def _func_high_dim():
            # Add a wrapper function to avoid parse the following code when len(shape) = 1
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_output: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_get_{state_id}"})

                storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                output = T.match_buffer(var_output, (batch_size, *shape), dtype)

                for i in range(batch_size):
                    for s in T.grid(*shape):
                        with T.block("copy"):
                            vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = history_slot_ids[vi]
                            # The following line is equivalent to:
                            # `output[vi, *vs] = storage[seq_id, history_id, *vs]`
                            # However, unpacking operator in subscript requires Python 3.11 or newer
                            T.buffer_store(
                                output, T.BufferLoad(storage, [seq_id, history_id, *vs]), [vi, *vs]
                            )

            return f

        return _func_one_dim() if len(shape) == 1 else _func_high_dim()

    @staticmethod
    def create_set_func(
        shape: Sequence[Union[int, tir.Var]],
        dtype: str,
        max_batch_size: Union[int, tir.Var],
        max_history: Union[int, tir.Var],
        state_id: int,
    ) -> tir.PrimFunc:
        """Create the set function with given state shape.

        Parameters
        ----------
        shape : Sequence[Union[int, tir.Var]]
            The shape of the state tensor.

        dtype: str
            The data type of the state tensor.

        max_batch_size : Union[int, tir.Var]
            The maximum batch size.

        max_history : Union[int, tir.Var]
            The maximum history length.

        state_id : int
            The id of the state, used for naming the function.

        Returns
        -------
        tir.PrimFunc
            The set function.
        """

        def _func_one_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_data: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_set_{state_id}"})

                storage = T.match_buffer(
                    var_storage, (max_batch_size, max_history, shape[0]), dtype
                )
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                data = T.match_buffer(var_data, (batch_size, shape[0]), dtype)

                for i in range(batch_size):
                    for s in range(shape[0]):
                        with T.block("copy"):
                            vi, vs = T.axis.remap("SS", [i, s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = (history_slot_ids[vi] + 1) % T.cast(
                                max_history, "int32"
                            )
                            storage[seq_id, history_id, vs] = data[vi, vs]

            return f

        def _func_high_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_data: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_set_{state_id}"})

                storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                data = T.match_buffer(var_data, (batch_size, *shape), dtype)

                for i in range(batch_size):
                    for s in T.grid(*shape):
                        with T.block("copy"):
                            vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = (history_slot_ids[vi] + 1) % T.cast(
                                max_history, "int32"
                            )
                            # The following line is equivalent to:
                            # `storage[seq_id, history_id, *vs] = data[vi, *vs]`
                            # However, unpacking operator in subscript requires Python 3.11 or newer
                            T.buffer_store(
                                storage, T.BufferLoad(data, [vi, *vs]), [seq_id, history_id, *vs]
                            )

            return f

        return _func_one_dim() if len(shape) == 1 else _func_high_dim()

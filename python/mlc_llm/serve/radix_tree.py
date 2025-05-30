"""The Paged Radix Tree class."""

from typing import List, Tuple, Union

import tvm
import tvm.ffi
from tvm.runtime import Object, ShapeTuple

from . import _ffi_api


@tvm.ffi.register_object("mlc.serve.PagedRadixTree")  # pylint: disable=protected-access
class PagedRadixTree(Object):
    """The paged radix tree to manage prefix and sequence."""

    def __init__(self):  # pylint: disable=super-init-not-called
        """
        Constructor of paged radix tree.
        """
        self.__init_handle_by_constructor__(_ffi_api.PagedRadixTree)  # type: ignore  # pylint: disable=no-member

    def match(self, tokens: Union[ShapeTuple, List, Tuple]) -> Tuple[int, ShapeTuple]:
        """
        Get all sequences with longest common prefix with given prefix tokens.

        Parameters
        ----------
        tokens : Union[ShapeTuple, List, Tuple]
            The prefix tokens for reference.

        Returns
        ------
        matched_offset : int
            The matched prefix length.
        seq_ids : ShapeTuple
            The array of matched sequence indice.
        """
        if isinstance(tokens, (list, tuple)):
            tokens = ShapeTuple(tokens)
        output = _ffi_api.PagedRadixTreeMatchPrefix(self, tokens)  # type: ignore  # pylint: disable=no-member
        if len(output) == 1:
            return output[0], []
        return output[0], output[1:]

    def add(self, seq_id: int) -> None:
        """
        Add an empty sequence.

        Parameters
        ----------
        seq_id : int
            The sequence ID for index.
        """
        _ffi_api.PagedRadixTreeAddSequence(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def remove(self, seq_id: int) -> None:
        """
        Remove a sequence.

        Parameters
        ----------
        seq_id : int
            The sequence ID to remove.
        """
        _ffi_api.PagedRadixTreeRemoveSequence(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def extend(self, seq_id: int, tokens: Union[ShapeTuple, List, Tuple]) -> None:
        """
        Extend a sequence with given tokens.

        Parameters
        ----------
        seq_id : int
            The sequence ID for index.
        tokens : Union[ShapeTuple, List, Tuple]
            The given tokens to extend.
        """
        if isinstance(tokens, (list, tuple)):
            tokens = ShapeTuple(tokens)
        _ffi_api.PagedRadixTreeExtendSequence(self, seq_id, tokens)  # type: ignore  # pylint: disable=no-member

    def rollback(self, seq_id: int, num_tokens: int) -> None:
        """
        Roll back a sequence by number of tokens.

        Parameters
        ----------
        seq_id : int
            The sequence ID for index.
        num_tokens : int
            The number of tokens to be rolled back.
        """
        _ffi_api.PagedRadixTreeRollBackSequence(self, seq_id, num_tokens)  # type: ignore  # pylint: disable=no-member

    def fork(self, seq_id: int, parent_seq_id: int, forked_offset: int) -> None:
        """
        Fork a sequence from parent sequence at given position.

        Parameters
        ----------
        seq_id : int
            The new sequence ID.
        parent_seq_id : int
            The parent sequence ID to fork from.
        forked_offset : int
            The position of parent sequence to fork at.
            The valid value is [1, length of forked sequence].
            If the position equals the length of forked sequence,
            the new sequence will copy the entire forked sequence.
        """
        _ffi_api.PagedRadixTreeForkSequence(self, seq_id, parent_seq_id, forked_offset)  # type: ignore  # pylint: disable=no-member

    def get(self, seq_id: int) -> ShapeTuple:
        """
        Get a sequence's all tokens.

        Parameters
        ----------
        seq_id : int
            The sequence ID for index.

        Returns
        ------
        tokens : ShapeTuple
            The sequence tokens.
        """
        return _ffi_api.PagedRadixTreeGetSequence(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def get_length(self, seq_id: int) -> int:
        """
        Get a sequence's length.

        Parameters
        ----------
        seq_id : int
            The sequence ID for index.

        Returns
        ------
        length : int
            The sequence length.
        """
        return _ffi_api.PagedRadixTreeGetSequenceLength(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def free_capacity(self) -> int:
        """
        Get the remaining token capacity of the paged radix tree.

        Returns
        ------
        capacity : int
            The remaining token capacity of the paged radix tree.
        """
        return _ffi_api.PagedRadixTreeFreeCapacity(self)  # type: ignore  # pylint: disable=no-member

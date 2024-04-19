import tvm
import tvm._ffi
from typing import Union, List, Tuple
from tvm.runtime import Object, ShapeTuple
from tvm.runtime.ndarray import NDArray

from . import _ffi_api


@tvm._ffi.register_object("mlc.serve.PagedRadixTree")  # pylint: disable=protected-access
class PagedRadixTree(Object):
    """The paged radix tree to manage prefix and sequence."""

    def __init__(self, num_pages: int, page_size: int, num_seqs: int):
        """
        Constructor of paged radix tree.

        Parameters
        ----------
        num_pages : int
            The number of radix tree pages.
        page_size : int
            The page size of each radix tree page.
        num_seqs : int
            The maximum number of sequence ID.
        """
        self.__init_handle_by_constructor__(_ffi_api.PagedRadixTree, num_pages, page_size, num_seqs)  # type: ignore  # pylint: disable=no-member

    def match(self, tokens: Union[ShapeTuple, List, Tuple]) -> Tuple[int, ShapeTuple]:
        """
        Get all sequences with longest common prefix with give prefix tokens.

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
        output = _ffi_api.PagedRadixTree_MatchPrefix(self, tokens)  # type: ignore  # pylint: disable=no-member
        if len(output) == 1:
            return output[0], []
        return output[0], output[1:]

    def add(self, seq_id: int):
        """
        Get all sequences with longest common prefix with give prefix tokens.

        Parameters
        ----------
        tokens : Union[ShapeTuple, List, Tuple]
            The prefix tokens for reference.
        """
        return _ffi_api.PagedRadixTree_AddSequence(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def remove(self, seq_id: int):
        """
        Remove a sequence.

        Parameters
        ----------
        seq_id : int
            The sequence ID to remove.
        """
        return _ffi_api.PagedRadixTree_RemoveSequence(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def extend(self, seq_id: int, tokens: Union[ShapeTuple, List, Tuple]):
        """
        Get all sequences with longest common prefix with give prefix tokens.

        Parameters
        ----------
        seq_id : int
            The sequence ID for index.
        tokens : Union[ShapeTuple, List, Tuple]
            The given tokens to extend.
        """
        if isinstance(tokens, (list, tuple)):
            tokens = ShapeTuple(tokens)
        return _ffi_api.PagedRadixTree_ExtendSequence(self, seq_id, tokens)  # type: ignore  # pylint: disable=no-member

    def fork(self, seq_id: int, parent_seq_id: int, forked_offset: int):
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
        return _ffi_api.PagedRadixTree_ForkSequence(self, seq_id, parent_seq_id, forked_offset)  # type: ignore  # pylint: disable=no-member

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
        return _ffi_api.PagedRadixTree_GetSequence(self, seq_id)  # type: ignore  # pylint: disable=no-member

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
        return _ffi_api.PagedRadixTree_GetSequenceLength(self, seq_id)  # type: ignore  # pylint: disable=no-member

    def free_capacity(self) -> int:
        """
        Get the remaining token capacity of the paged radix tree.

        Returns
        ------
        capacity : int
            The remaining token capacity of the paged radix tree.
        """
        return _ffi_api.PagedRadixTree_FreeCapacity(self)  # type: ignore  # pylint: disable=no-member

from tvm import TVMError
from tvm.runtime import ShapeTuple
from mlc_llm.serve import PagedRadixTree


def test_add():
    prt = PagedRadixTree(16, 128, 16)
    prt.add(0)
    assert prt.get(0) == []


def test_remove():
    prt = PagedRadixTree(32, 128, 16)
    capacity = prt.free_capacity()
    prt.add(0)
    prt.remove(0)
    prt.add(0)
    prt.extend(0, [1 for _ in range(200)])
    prt.remove(0)
    assert prt.free_capacity() == capacity

    prt.add(1)
    prt.extend(1, [1 for _ in range(200)])
    capacity = prt.free_capacity()
    prt.add(2)
    prt.extend(2, [1 for _ in range(100)] + [2 for _ in range(100)])
    prt.remove(2)
    assert prt.free_capacity() == capacity

    prt.add(3)
    prt.extend(3, [1 for _ in range(200)])
    prt.remove(3)
    assert prt.free_capacity() == capacity


def test_extend():
    prt = PagedRadixTree(1024, 256, 256)
    L = prt.free_capacity() // 1024
    H = L // 2
    Q = L // 4
    seq_id = 0
    for start_pos in [0, H, L, L + H]:
        for length in [Q, L - H, L, 2 * L - H, 2 * L]:
            prt.add(seq_id)
            if start_pos:
                tokens_1 = [seq_id for _ in range(start_pos)]
                prt.extend(seq_id, tokens_1)
                assert prt.get(seq_id) == tokens_1
            else:
                tokens_1 = []
            tokens_2 = [seq_id for _ in range(length)]
            prt.extend(seq_id, tokens_2)
            assert prt.get(seq_id) == tokens_1 + tokens_2
            seq_id += 1


def test_fork():
    prt = PagedRadixTree(1024, 256, 256)
    L = prt.free_capacity() // 1024
    H = L // 2
    Q = L // 4
    seq_id = 0
    length_list = [Q, H, L, L + Q, L + H, L * 2]
    for p_idx in range(1, len(length_list)):
        for c_idx in range(0, p_idx + 1):
            prt.add(seq_id)
            tokens = [seq_id for _ in range(length_list[p_idx])]
            prt.extend(seq_id, tokens)
            prt.fork(seq_id + 1, seq_id, length_list[c_idx])
            assert prt.get(seq_id + 1) == tokens[: length_list[c_idx]]
            seq_id += 2


if __name__ == "__main__":
    test_add()
    test_remove()
    test_extend()
    test_fork()

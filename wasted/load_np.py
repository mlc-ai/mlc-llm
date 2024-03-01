import numpy as np

datas = []
pre_x = None


def concat_qkv(datas):
    qs = []
    ks = []
    vs = []
    for d in datas:
        print(d.shape)
        qkv = np.split(d, 3, axis=-1)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        qs.append(q)
        ks.append(k)
        vs.append(v)

    result = qs + ks + vs
    return np.concatenate(result, -1)


for i in range(4):
    x = np.load(f"disco_test/{i}.npy")
    datas.append(x)
    sum_result = np.sum(datas, axis=0)
    mean_result = np.mean(datas, axis=0)
    # concat_result = concat_qkv(datas)
    concat_result = np.concatenate(datas, axis=-1)

    # print(i)
    # print(x)
    # print("====")
    # if pre_x is None or not np.allclose(x, pre_x):
    #     print(i)
    #     print(x)
    pre_x = x

std_x = np.load("disco_test/single.npy")


result = sum_result
# print(datas[0])
# print(sum_result)
# print(concat_result)
# print(concat_result.shape)
# print(std_x.shape)
print(result)
print(std_x)
# print(concat_result - std_x)
print(np.max(result - std_x))
# print(np.sum((concat_result - std_x) > 0))
# print(np.argmax(concat_result - std_x))
print(np.allclose(result, std_x, atol=1e-1))

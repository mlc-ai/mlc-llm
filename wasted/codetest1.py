# pylint: skip-file
cnt = 0


def bt(nums, target, cur_i, cur_v):
    global cnt
    if cur_i == len(nums):
        if cur_v == target:
            cnt = cnt + 1
        else:
            pass
    else:
        bt(nums, target, cur_i + 1, cur_v + nums[cur_i])
        bt(nums, target, cur_i + 1, cur_v - nums[cur_i])


def f(nums, target):
    bt(nums, target, cur_i=0, cur_v=0)
    return cnt


nums = [1, 1, 1, 1, 1]
target = 3
print(f(nums, target))

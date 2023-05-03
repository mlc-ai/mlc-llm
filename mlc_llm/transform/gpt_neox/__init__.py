def lookup(func):
    from . import dolly_v2_3b

    ret = dolly_v2_3b.lookup(func)
    if ret is not None:
        return ret
    return None

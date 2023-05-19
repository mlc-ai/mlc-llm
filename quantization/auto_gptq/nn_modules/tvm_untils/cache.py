import os
import json
from .pycuda_warpper import TVMHandler, TVMExecutable


handler_database = {}
cache_dir = ".cache"
cache_file = "handler_database.json"

faster_cache = {}
def get_handler(bits: int, n: int, k: int, group_size: int = -1):
    key = f"b{bits}n{n}k{k}"
    key += "" if group_size == -1 else f"g{group_size}"
    print(key)
    if key in faster_cache:
        return faster_cache[key]
    else:
        # Check if the cache folder exists, create it if not
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Check if the cache file exists, read the data if it does
        if os.path.isfile(os.path.join(cache_dir, cache_file)):
            with open(os.path.join(cache_dir, cache_file), "r") as f:
                handler_database = json.load(f)
                # check if the key exists, if it does, load the handler from the cache file
                if key in handler_database:
                    # print finds the key
                    print("finds the key ", key, " in the cache file ", cache_file)
                    handler = TVMHandler(bits=bits, n=n, k=k, group_size=group_size, load_from_cache=True)
                    for candidate in handler.m_candidates:
                        func_name = handler_database[key][f"m{candidate}"]["func_name"]
                        code = handler_database[key][f"m{candidate}"]["code"]
                        executable = TVMExecutable(src=code, name=func_name)
                        params = handler_database[key][f"m{candidate}"]["params"]
                        setattr(handler, f"m{candidate}", executable)
        else:
            handler_database = {}
        
        # If the key doesn't exist, create it and save it to the cache file
        if key not in handler_database:
            print("doesn't find the key ", key, " in the cache file ", cache_file)
            handler = TVMHandler(bits=bits, n=n, k=k, group_size=group_size, load_from_cache=False)
            candidates = handler.m_candidates
            _dump = {}
            for candidate in candidates:
                _c = f"m{candidate}"
                executable = getattr(handler, _c)
                _dump[_c] = {
                    "func_name": executable.func_name,
                    "code": executable.source_code,
                    "params": handler.configurations[_c]
                }
                
            handler_database[key] = _dump
            with open(os.path.join(cache_dir, cache_file), "w") as f:
                json.dump(handler_database, f, indent=4)
        
        faster_cache[key] = handler
    
    return handler

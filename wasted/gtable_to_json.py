with open("sheet.csv", "r") as f:
    lines = f.readlines()

keys = lines[0].strip().split(",")
dd = {}
print(keys)
for line in lines[1:]:
    values = line.strip().split(",")
    assert len(values) == len(keys)
    model_name = values[0]
    dd[model_name] = {}
    for k, v in zip(keys[1:], values[1:]):
        dd[model_name][k] = v

import json

with open("model_info.json", "w+") as f:
    json.dump(dd, f)

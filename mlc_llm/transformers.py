# pylint: disable=import-outside-toplevel
def _get_model_worker(_args) -> None:
    import json
    import os

    import numpy as np
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    model: str
    dump_path: str
    model, dump_path = _args
    config_path = os.path.join(dump_path, "config.json")
    if os.path.exists(config_path):
        print("Model weights already exist under:", dump_path)
        return

    print("Extracting weights for model:", model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
    )
    params = [
        (
            name,
            param.detach().cpu().numpy(),
        )
        for name, param in hf_model.named_parameters()
    ]
    del hf_model

    os.makedirs(dump_path, exist_ok=True)
    for i, (name, param) in enumerate(params):
        param_path = os.path.join(dump_path, f"param_{i}.npy")
        np.save(param_path, param)

    with open(config_path, "w", encoding="utf-8") as o_f:
        json.dump(
            [name for name, _ in params],
            o_f,
        )
    print("Model weights dumped to:", dump_path)


def get_model(model: str, dump_path: str):
    import json
    import multiprocessing
    import os
    from typing import List, Tuple

    import numpy as np
    from tqdm import tqdm

    with multiprocessing.Pool(processes=1) as pool:
        result = pool.map(
            _get_model_worker,
            [
                (model, dump_path),
            ],
        )
    print("Loading model weights from:", dump_path)
    config_path = os.path.join(dump_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as i_f:
        config = json.load(i_f)
    param_dict: List[Tuple[str, np.ndarray]] = []
    for i, name in tqdm(enumerate(config), total=len(config)):
        param_path = os.path.join(dump_path, f"param_{i}.npy")
        param_dict.append((name, np.load(param_path)))
    print("Loading done")
    return param_dict

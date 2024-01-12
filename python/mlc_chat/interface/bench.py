"""Python entrypoint of benchmark."""
from typing import Optional

from mlc_chat.chat_module import ChatConfig, ChatModule

from .chat import ChatConfigOverride


def bench(  # pylint: disable=too-many-arguments
    model: str,
    prompt: str,
    device: str,
    opt: str,
    overrides: ChatConfigOverride,
    generate_length: int,
    model_lib_path: Optional[str],
):
    """run the benchmarking"""
    # Set up chat config
    config = ChatConfig(opt=opt)
    # Apply overrides
    config = overrides.apply(config)
    # Set up ChatModule
    cm = ChatModule(model, device, chat_config=config, model_lib_path=model_lib_path)

    output = cm.benchmark_generate(prompt, generate_length=generate_length)
    print(f"Generated text:\n{output}\n")
    print(f"Statistics:\n{cm.stats(verbose=True)}")

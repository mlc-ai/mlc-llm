import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from .types import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)


@dataclass
class EngineArgs:
    """Arguments for vLLM engine."""
    model: str
    lib_path: str
    device: str
    # tokenizer: Optional[str] = None
    # tokenizer_mode: str = 'auto'
    # trust_remote_code: bool = False
    # download_dir: Optional[str] = None
    # load_format: str = 'auto'
    # dtype: str = 'auto'
    random_seed: int = 0
    #TODO(amalyshe): this number is taken randomly, need initialize from model config by default
    max_model_len: [int] = 4096
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256

    # def __post_init__(self):
    #     if self.tokenizer is None:
    #         self.tokenizer = self.model

    @staticmethod
    def add_cli_args(
            parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Shared CLI arguments for vLLM engine."""
        # Model arguments
        parser.add_argument(
            '--model',
            type=str,
            help =
                """
                The model folder after compiling with MLC-LLM build process. The parameter
                can either be the model name with its quantization scheme
                (e.g. ``Llama-2-7b-chat-hf-q4f16_1``), or a full path to the model
                folder. In the former case, we will use the provided name to search
                for the model folder over possible paths.
                """
            )
        parser.add_argument(
            '--lib-path',
            default=None,
            help =
                    """
                    The full path to the model library file to use (e.g. a ``.so`` file).
                    """
        )
        parser.add_argument(
            '--device',
            default = "auto",
            help =
                """
                The description of the device to run on. User should provide a string in the
                form of 'device_name:device_id' or 'device_name', where 'device_name' is one of
                'cuda', 'metal', 'vulkan', 'rocm', 'opencl', 'auto' (automatically detect the
                local device), and 'device_id' is the device id to run on. If no 'device_id'
                is provided, it will be set to 0 by default.
                """
        )
        parser.add_argument('--max-model-len',
                            type=int,
                            default=EngineArgs.max_model_len,
                            help='model context length. If unspecified, '
                            'will be automatically derived from the model.')
        # Parallel arguments
        parser.add_argument('--pipeline-parallel-size',
                            '-pp',
                            type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='number of pipeline stages')
        parser.add_argument('--tensor-parallel-size',
                            '-tp',
                            type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='number of tensor parallel replicas')
        # KV cache arguments
        parser.add_argument('--block-size',
                            type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='token block size')
        parser.add_argument('--random-seed',
                            type=int,
                            default=EngineArgs.random_seed,
                            help=
                                """
                                The random seed to initialize all the RNG used in mlc-chat. By default,
                                no seed is set.
                                """
                            )
        parser.add_argument('--swap-space',
                            type=int,
                            default=EngineArgs.swap_space,
                            help='CPU swap space size (GiB) per GPU')
        parser.add_argument('--gpu-memory-utilization',
                            type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='the percentage of GPU memory to be used for'
                            'the model executor')
        parser.add_argument('--max-num-batched-tokens',
                            type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='maximum number of batched tokens per '
                            'iteration')
        parser.add_argument('--max-num-seqs',
                            type=int,
                            default=EngineArgs.max_num_seqs,
                            help='maximum number of sequences per iteration')
        parser.add_argument('--disable-log-stats',
                            action='store_true',
                            help='disable logging statistics')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineArgs':
        # Get the list of attributes of this dataclass.
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # Set the attributes from the parsed arguments.
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        model_config = ModelConfig(self.model, self.lib_path, self.device, self.random_seed,
                                   self.max_model_len)
        cache_config = CacheConfig(self.block_size, self.gpu_memory_utilization, self.swap_space)
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size)
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs,
                                           model_config.max_model_len)
        return model_config, cache_config, parallel_config, scheduler_config

"""For testing the functionality of `BuildArgs` and `convert_build_args_to_argparser`."""
import argparse
import dataclasses
import unittest

from mlc_llm import BuildArgs, core, utils


def old_make_args():
    """The exact old way of creating `ArgumentParser`, used to test whether
    `BuildArgs` is equivalent to this."""
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="auto",
        help=(
            'The name of the model to build. If it is "auto", we will '
            'automatically set the model name according to "--model-path", '
            '"hf-path" or the model folders under "--artifact-path/models"'
        ),
    )
    args.add_argument(
        "--hf-path",
        type=str,
        default=None,
        help="Hugging Face path from which to download params, tokenizer, and config",
    )
    args.add_argument(
        "--quantization",
        type=str,
        choices=[*utils.quantization_schemes.keys()],
        default=list(utils.quantization_schemes.keys())[0],
        help="The quantization mode we use to compile.",
    )
    args.add_argument(
        "--max-seq-len",
        type=int,
        default=-1,
        help="The maximum allowed sequence length for the model.",
    )
    args.add_argument(
        "--target", type=str, default="auto", help="The target platform to compile the model for."
    )
    args.add_argument(
        "--reuse-lib",
        type=str,
        default=None,
        help="Whether to reuse a previously generated lib.",
    )
    args.add_argument(
        "--artifact-path", type=str, default="dist", help="Where to store the output."
    )
    args.add_argument(
        "--use-cache",
        type=int,
        default=1,
        help="Whether to use previously pickled IRModule and skip trace.",
    )
    args.add_argument(
        "--debug-dump",
        action="store_true",
        default=False,
        help="Whether to dump debugging files during compilation.",
    )
    args.add_argument(
        "--debug-load-script",
        action="store_true",
        default=False,
        help="Whether to load the script for debugging.",
    )
    args.add_argument(
        "--llvm-mingw",
        type=str,
        default="",
        help="/path/to/llvm-mingw-root, use llvm-mingw to cross compile to windows.",
    )
    args.add_argument(
        "--system-lib", action="store_true", default=False, help="A parameter to `relax.build`."
    )
    args.add_argument(
        "--sep-embed",
        action="store_true",
        default=False,
        help=(
            "Build with separated embedding layer, only applicable to LlaMa. "
            "This feature is in testing stage, and will be formally replaced after "
            "massive overhaul of embedding feature for all models and use cases"
        ),
    )

    return args


# Referred to HfArgumentParserTest from https://github.com/huggingface/
# transformers/blob/e84bf1f734f87aa2bedc41b9b9933d00fc6add98/tests/utils
# /test_hf_argparser.py#L143
class BuildArgsTest(unittest.TestCase):
    """Tests whether BuildArgs reaches parity with regular ArgumentParser."""

    def argparsers_equal(self, parse_a: argparse.ArgumentParser, parse_b: argparse.ArgumentParser):
        """
        Small helper to check pseudo-equality of parsed arguments on `ArgumentParser` instances.
        """
        self.assertEqual(
            len(parse_a._actions), len(parse_b._actions)
        )  # pylint: disable=protected-access
        for x, y in zip(parse_a._actions, parse_b._actions):  # pylint: disable=protected-access
            xx = {k: v for k, v in vars(x).items() if k != "container"}
            yy = {k: v for k, v in vars(y).items() if k != "container"}
            # Choices with mixed type have custom function as "type"
            # So we need to compare results directly for equality
            if xx.get("choices", None) and yy.get("choices", None):
                for expected_choice in yy["choices"] + xx["choices"]:
                    self.assertEqual(xx["type"](expected_choice), yy["type"](expected_choice))
                del xx["type"], yy["type"]

            self.assertEqual(xx, yy)

    def test_new_and_old_arg_parse_are_equivalent(self):
        """Tests whether creating `ArgumentParser` from `BuildArgs` is equivalent
        to the conventional way of creating it."""
        self.argparsers_equal(core.convert_build_args_to_argparser(), old_make_args())

    def test_namespaces_are_equivalent_str(self):
        """Tests whether the resulting namespaces from command line entry
        and Python API entry are equivalent, as they are passed down to the
        same workflow."""
        # Namespace that would be created through Python API build_model
        build_args = BuildArgs(model="RedPJ", target="cuda")
        build_args_as_dict = dataclasses.asdict(build_args)
        build_args_namespace = argparse.Namespace(**build_args_as_dict)

        # Namespace that would be created through commandline
        empty_args = core.convert_build_args_to_argparser()
        parsed_args = empty_args.parse_args(["--model", "RedPJ", "--target", "cuda"])

        self.assertEqual(build_args_namespace, parsed_args)

        # Modify build_args so that it would not be equivalent
        build_args = BuildArgs(model="RedPJ", target="vulkan")
        build_args_as_dict = dataclasses.asdict(build_args)
        build_args_namespace = argparse.Namespace(**build_args_as_dict)

        self.assertNotEqual(build_args_namespace, parsed_args)

    def test_namespaces_are_equivalent_str_boolean_int(self):
        """Same test, but for a mixture of argument types."""
        # 1. Equal
        build_args = BuildArgs(model="RedPJ", max_seq_len=20, debug_dump=True)
        build_args_as_dict = dataclasses.asdict(build_args)
        build_args_namespace = argparse.Namespace(**build_args_as_dict)

        # Namespace that would be created through commandline
        empty_args = core.convert_build_args_to_argparser()
        parsed_args = empty_args.parse_args(
            ["--model", "RedPJ", "--max-seq-len", "20", "--debug-dump"]
        )
        self.assertEqual(build_args_namespace, parsed_args)

        # 2. Not equal - missing boolean
        build_args = BuildArgs(model="RedPJ", max_seq_len=20)
        build_args_as_dict = dataclasses.asdict(build_args)
        build_args_namespace = argparse.Namespace(**build_args_as_dict)
        self.assertNotEqual(build_args_namespace, parsed_args)

        # 3. Not equal - different integer
        build_args = BuildArgs(model="RedPJ", max_seq_len=18, debug_dump=True)
        build_args_as_dict = dataclasses.asdict(build_args)
        build_args_namespace = argparse.Namespace(**build_args_as_dict)
        self.assertNotEqual(build_args_namespace, parsed_args)


if __name__ == "__main__":
    unittest.main()

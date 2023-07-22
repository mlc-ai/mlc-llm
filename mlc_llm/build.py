"""Script for building/compiling models."""
from mlc_llm import core

def main():
    """Main method for building model from command line."""
    empty_args = core.convert_build_args_to_argparser()  # Create new ArgumentParser
    parsed_args = empty_args.parse_args()  # Parse through command line
    # Post processing of arguments
    parsed_args = core._parse_args(parsed_args)  # pylint: disable=protected-access
    core.build_model_from_args(parsed_args)

if __name__ == "__main__":
    main()

"""Script for building/compiling models."""
import contextlib
import sys

import tvm
from mlc_llm import core
from tvm.relax.ir.instrument import WellFormedInstrument
from mlc_llm.utils import DumpBeforeAfter


@contextlib.contextmanager
def debug_on_except():
    try:
        yield
    finally:
        if sys.exc_info() == (None, None, None):
            return

        import traceback

        try:
            import ipdb as pdb
        except ImportError:
            import pdb

        traceback.print_exc()
        pdb.post_mortem()


def main():
    """Main method for building model from command line."""
    empty_args = core.convert_build_args_to_argparser()  # Create new ArgumentParser
    parsed_args = empty_args.parse_args()  # Parse through command line

    with contextlib.ExitStack() as stack:
        # Enter an exception-catching context before post-processing
        # the arguments, in case the post-processing itself raises an
        # exception.
        if parsed_args.pdb:
            stack.enter_context(debug_on_except())

        # Post processing of arguments
        parsed_args = core._parse_args(parsed_args)  # pylint: disable=protected-access

        # Enter any additional debug contexts
        instruments = []
        if parsed_args.debug_dump:
            instruments.append(DumpBeforeAfter(parsed_args, excludes_list=['tir']))   
            
        if parsed_args.assert_well_formed:
            instruments.append(WellFormedInstrument())  
            
        if instruments:
            stack.enter_context(tvm.transform.PassContext(instruments=instruments))

        core.build_model_from_args(parsed_args)


if __name__ == "__main__":
    main()

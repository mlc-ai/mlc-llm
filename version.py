# pylint: disable=missing-docstring
import argparse
import logging
import os
import subprocess

# Modify the following value during release
# ---------------------------------------------------
# Current version:
# We use the version of the incoming release for code
# that is under development.
#
# It is also fallback version to be used when --git-describe
# is not invoked, or when the repository does not present the
# git tags in a format that this script can use.
#
# Two tag formats are supported:
# - vMAJ.MIN.PATCH (e.g. v0.8.0) or
# - vMAJ.MIN.devN (e.g. v0.8.dev0)

# ---------------------------------------------------

__version__ = "0.1.dev0"
PROJ_ROOT = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))


def py_str(cstr):
    return cstr.decode("utf-8")


def git_describe_version():
    """Get PEP-440 compatible public and local version using git describe.

    Returns
    -------
    pub_ver: str
        Public version.

    local_ver: str
        Local version (with additional label appended to pub_ver).

    Notes
    -----
    - We follow PEP 440's convention of public version
      and local versions.
    - Only tags conforming to vMAJOR.MINOR.REV (e.g. "v0.7.0")
      are considered in order to generate the version string.
      See the use of `--match` in the `git` command below.

    Here are some examples:

    - pub_ver = '0.7.0', local_ver = '0.7.0':
      We are at the 0.7.0 release.
    - pub_ver =  '0.8.dev94', local_ver = '0.8.dev94+g0d07a329e':
      We are at the 0.8 development cycle.
      The current source contains 94 additional commits
      after the most recent tag(v0.7.0),
      the git short hash tag of the current commit is 0d07a329e.
    """
    cmd = [
        "git",
        "describe",
        "--tags",
        "--match",
        "v[0-9]*.[0-9]*.[0-9]*",
        "--match",
        "v[0-9]*.[0-9]*.dev[0-9]*",
    ]
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=PROJ_ROOT,
    ) as proc:
        (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = py_str(out)
        logging.warning("git describe: %s", msg)
        return None, None
    describe = py_str(out).strip()
    arr_info = describe.split("-")

    # Remove the v prefix, mainly to be robust
    # to the case where v is not presented as well.
    if arr_info[0].startswith("v"):
        arr_info[0] = arr_info[0][1:]

    # hit the exact tag
    if len(arr_info) == 1:
        return arr_info[0], arr_info[0]

    if len(arr_info) != 3:
        logging.warning("Invalid output from git describe %s", describe)
        return None, None

    dev_pos = arr_info[0].find(".dev")

    # Development versions:
    # The code will reach this point in case it can't match a full release version, such as v0.7.0.
    #
    # 1. in case the last known label looks like vMAJ.MIN.devN e.g. v0.8.dev0, we use
    # the current behavior of just using vMAJ.MIN.devNNNN+gGIT_REV
    if dev_pos != -1:
        dev_version = arr_info[0][: arr_info[0].find(".dev")]
    # 2. in case the last known label looks like vMAJ.MIN.PATCH e.g. v0.8.0
    # then we just carry on with a similar version to what git describe provides, which is
    # vMAJ.MIN.PATCH.devNNNN+gGIT_REV
    else:
        dev_version = arr_info[0]

    pub_ver = f"{dev_version}.dev{arr_info[1]}"
    local_ver = f"{pub_ver}+{arr_info[2]}"
    return pub_ver, local_ver


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Detect and synchronize version.")
    parser.add_argument(
        "--print-version",
        action="store_true",
        help="Print version to the command line. No changes is applied to files.",
    )
    parser.add_argument(
        "--git-describe",
        action="store_true",
        help="Use git describe to generate development version.",
    )
    parser.add_argument("--dry-run", action="store_true")
    pub_ver, local_ver = git_describe_version()
    opt = parser.parse_args()
    pub_ver, local_ver = None, None
    if opt.git_describe:
        pub_ver, local_ver = git_describe_version()
    if pub_ver is None:
        pub_ver = __version__
    if local_ver is None:
        local_ver = __version__
    if opt.print_version:
        print(local_ver)


if __name__ == "__main__":
    main()

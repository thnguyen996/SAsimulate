from argparse import ArgumentParser
from sasimulate import __version__

def cli(args=None):
    p = ArgumentParser(
        description="Simulate stuck-at-faults neural network ",
        conflict_handler='resolve'
    )
    p.add_argument(
        '-V', '--version',
        action='version',
        help='Show the conda-prefix-replacement version number and exit.',
        version="sasimulate %s" % __version__,
    )

    args = p.parse_args(args)

    # do something with the args
    print("CLI template - fix me up!")

    # No return value means no error.
    # Return a value of 1 or higher to signify an error.
    # See https://docs.python.org/3/library/sys.html#sys.exit


if __name__ == '__main__':
    import sys
    cli(sys.argv[1:])

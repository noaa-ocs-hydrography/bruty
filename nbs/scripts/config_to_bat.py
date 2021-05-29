""" Script to make a batch file with set statements based on the config values supplied.
Using a straight  batch script will not use the python processing needed for the advanced config syntax (like 50{utm}).
Also can't export the value from python as Windows limits environment variables to the current process and children,
so while setting os.environ will work in subprocess.call or popen it doesn't propogate back to the calling batch file.

"""

import os
import sys
import argparse
import subprocess

from nbs.configs import get_logger, iter_configs, set_stream_logging, log_config, parse_multiple_values



def make_parser():
    parser = argparse.ArgumentParser(description='Read a NBS ini config and put the requested section/variable into the environment variables')
    parser.add_argument("-?", "--show_help", action="store_true",
                        help="show this help message and exit")

    parser.add_argument("-s", "--section", type=str, metavar='section', default="DEFAULT",
                        help="Section name will read [DEFAULT] if not supplied")
    parser.add_argument("-v", "--variable", type=str, action='append', metavar='variable', default=[],  # nargs="+"
                        help="variable to read from the ini section (can specify more than one)")
    parser.add_argument("-i", "--inifile", type=str, metavar='inifile', default="",
                        help="path to ini file")
    parser.add_argument("-o", "--output", type=str, metavar='output', default="",
                        help="path to ini file")
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # if len(sys.argv[1:]) == 0:
    #     parser.print_help()
    args = parser.parse_args()
    if args.show_help or not args.inifile or not args.variable:
        parser.print_help()
        sys.exit()

    if args.output:
        out_batch = open(args.output, "w")
        # out_batch.write("echo Setting variables from config into environment\n")
        for config_filename, config_file in iter_configs([args.inifile]):
            config = config_file[args.section]
            for varname in args.variable:
                var = config[varname]
                os.environ[varname] = var
                line = f"set {varname}={os.environ[varname]}\n"  # don't use spaces in a set command!
                print(line)
                out_batch.write(line)
        out_batch.close()

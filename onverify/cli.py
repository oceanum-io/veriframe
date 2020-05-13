# -*- coding: utf-8 -*-

"""Console script for ondata."""
import importlib
import logging
import sys

import click
import yaml

from oncore.date import timedelta

DFMTS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d.%H%M%S",
    "%Y%m%d",
    "%Y-%m",
    "%Y%m",
]


def import_pycallable(pycallable):
    pycallable = pycallable.split(".")
    method = pycallable[-1]
    module_str = ".".join(pycallable[:-1])
    module = importlib.import_module(module_str)
    return getattr(module, method)


@click.group()
def main():
    pass


@main.command()
@click.argument("pycallable", type=str)
@click.argument("args", type=list, default=[])
@click.option(
    "-s", "--start", help="Start date", type=click.DateTime(formats=DFMTS),
)
@click.option(
    "-e", "--end", help="Start date", type=click.DateTime(formats=DFMTS), default=None
)
@click.option("-f", "--freq", help="Frequency", type=str, default=None)
@click.option(
    "-m",
    "--methods",
    help="List of methods in object to run",
    default=["__call__"],
    show_default=True,
)
@click.option(
    "--kwargs",
    "-k",
    multiple=True,
    help="additional key value pairs in the format key:value",
)
def satellite(pycallable, start, end, freq, args, kwargs, methods):
    kw = {}
    for item in kwargs:
        split = item.split(":", 1)
        if isinstance(split[1], str):
            try:
                kw.update({split[0]: float(split[1])})
            except Exception as e:
                if "," in split[1]:
                    kw.update({split[0]: split[1].split(",")})
                else:
                    kw.update({split[0]: split[1]})
        else:
            kw.update({split[0]: split[1]})
    dateerror = "Both end and freq specified, specify one only"
    if freq:
        if end:
            raise Exception("Both end and freq specified, specify one only")
        else:
            end = start + timedelta(freq)
    else:
        if not end:
            raise Exception("Neither end or freq specified please specify one")

    kw.update(dict(start=start, end=end))

    module = import_pycallable(pycallable)
    instance = module(*args, **kw)
    for method in methods:
        getattr(instance, method)()


if __name__ == "__main__":
    main()

import click

from .labfront import commands as labfront


@click.group()
def cli():
    pass


cli.add_command(labfront.labfront)

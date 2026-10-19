import click

from .conversion import commands as conversion_commands


@click.group()
def cli():
    pass


cli.add_command(conversion_commands.convert)

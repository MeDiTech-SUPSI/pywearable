import click

from .conversion import commands as conversion_commands

# from .labfront import commands as labfront


@click.group()
def cli():
    pass


# cli.add_command(labfront.labfront)
cli.add_command(conversion_commands.convert)

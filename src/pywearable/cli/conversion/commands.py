import click as cli

from .lifesnaps import lifesnaps_to_parquet
from .tesserae import tesserae_to_parquet


@cli.group()
def convert():
    pass


convert.add_command(lifesnaps_to_parquet)
convert.add_command(tesserae_to_parquet)

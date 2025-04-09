"""Command-line interface for nwbinfo."""

import sys
import click
from .core import get_nwbfile_usage_script

@click.group()
def main():
    """CLI tool to analyze NWB files."""
    pass

@main.command()
@click.argument('url')
def usage_script(url):
    """Generate Python code to access NWB file objects and fields.

    URL: Path or URL to the NWB file.

    Example: get-nwbfile-info usage-script https://api.dandiarchive.org/api/assets/7423831f-100c-4103-9dde-73ac567d32fb/download/
    """
    try:
        result = get_nwbfile_usage_script(url)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

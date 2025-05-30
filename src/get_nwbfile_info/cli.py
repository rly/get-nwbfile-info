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
@click.option('--output', '-o', type=click.Path(writable=True), help='Output file path. If not provided, prints to stdout.')
def usage_script(url, output):
    """Generate Python code to access NWB file objects and fields.

    URL: Can be one of:
    - Local file path to an NWB file
    - Direct URL to an NWB file
    - DANDI archive reference (format: DANDI:[ID]:[VERSION]:[path])
    - .lindi.json or .lindi.tar file path/URL

    Examples:
    # Direct DANDI URL
    get-nwbfile-info usage-script https://api.dandiarchive.org/api/assets/7423831f-100c-4103-9dde-73ac567d32fb/download/

    # DANDI archive reference
    get-nwbfile-info usage-script DANDI:001349:0.250520.1729:sub-C57-C2-2-AL/sub-C57-C2-2-AL_ses-2_ophys.nwb

    # Local NWB file
    get-nwbfile-info usage-script path/to/file.nwb

    # With output file
    get-nwbfile-info usage-script https://api.dandiarchive.org/.../download/ -o output.py
    """
    try:
        result = get_nwbfile_usage_script(url)
        if output:
            with open(output, 'w') as f:
                f.write(result)
            print(f"Output written to {output}")
        else:
            print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

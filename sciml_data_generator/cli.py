"""Console script for sciml_data_generator."""
import sys
import click
import os
import numpy as np
import time

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")

def load_mesh_files(file_dir, base_file):
    # read the no. of nodes and cells
    f      = open(file_dir+base_file+'.1.node', 'r')
    line   = f.readline()
    nnodes = int(line.split()[0])
    
    f.close()
    f      = open(file_dir+base_file+'.1.ele', 'r')
    line   = f.readline()
    ncels  = int(line.split()[0])
    
    f.close()
    # read the nodes, cells and the density of the cells
    nodes = np.loadtxt(file_dir+base_file+'.1.node', usecols=(1,2,3),   skiprows=1, max_rows=nnodes, dtype=float)
    eles  = np.loadtxt(file_dir+base_file+'.1.ele', usecols=(1,2,3,4), skiprows=1, max_rows=ncels , dtype=int)
    eles  = eles - 1
    return nnodes, ncels, nodes, eles

@click.command()
@click.option(
    "--mesh_dir",
    type=click.Path(),
    help="Directory where mesh files are located",
    show_default=True,
)
@click.option(
    "--base_file_name",
    type=str,
    help="Base file name",
    show_default=True,
)
@click.option(
    "--mdl_dir",
    type=click.Path(),
    help="Model directory",
    show_default=True,
)
@click.option(
    "--sus_name_prefix",
    type=str,
    help="Susceptibility file name prefix",
    show_default=True,
)
@click.option(
    "--kx_name_prefix", type=str, help="KX file name prefix", show_default=True
)
@click.option(
    "--ky_name_prefix", type=str, help="KY file name prefix", show_default=True
)
@click.option(
    "--kz_name_prefix", type=str, help="KZ file name prefix", show_default=True
)
@click.option(
    "--data_file",
    type=click.Path(),
    help="Receiver locations file in CSV format",
    show_default=True,
)
@click.option(
    "--results_dir",
    type=click.Path(),
    help="Results directory",
    show_default=True,
)
@click.option(
    "--results_file_prefix", type=str, help="Results file prefix", show_default=True
)
@click.option(
    "--n_cpu",
    type=int,
    default=1,
    help="Number of CPU cores for computation",
    show_default=True,
)
@click.option(
    "--n_samples",
    type=int,
    default=12,
    help="Number of CPU cores for computation",
    show_default=True,
)
@click.option(
    "--ismag",
    is_flag=True,
    default=True,
    help="TBD",
    show_default=True,
)
@click.option(
    "--istensor",
    is_flag=True,
    default=False,
    help="TBD",
    show_default=True,
)
@click.option(
    "--Bx",
    type=float,
    default=4594.8,
    help="Ambient magnetic flux Bx",
    show_default=True,
)
@click.option(
    "--By",
    type=float,
    default=19887.1,
    help="Ambient magnetic flux By",
    show_default=True,
)
@click.option(
    "--Bz",
    type=float,
    default=41568.2,
    help="Ambient magnetic flux Bz",
    show_default=True,
)
def main(**kwargs):
    """Console script for sciml_data_generator."""
    click.echo(
        "Replace this message by putting your code into "
        "sciml_data_generator.cli.main"
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")
    print(kwargs)
    
    bx = kwargs["bx"]
    by = kwargs["by"]
    bz = kwargs["bz"]
    mesh_dir = kwargs["mesh_dir"]
    base_file_name = kwargs["base_file_name"]
    nverts, ntets, verts, tetra = load_mesh_files(mesh_dir, base_file_name)

    bv = np.sqrt(bx**2 + by**2 + bz**2)
    
    LX = np.float32(bx/bv)
    LY = np.float32(by/bv)
    LZ = np.float32(bz/bv)

    
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

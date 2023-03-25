"""Console script for sciml_data_generator."""
import sys
import click
import os
import numpy as np
import time

"""base_file_name = 'Hollister_Test_c_5m'

# model npy files Hollister_Test_c_5m
mdl_dir = '../Input_Model_npy_files/'
sus_name = 'Pipe_Model_samp_sus_'
kx_name = 'Pipe_Model_samp_kx_'
ky_name = 'Pipe_Model_samp_ky_'
kz_name = 'Pipe_Model_samp_kz_'

# Observation points location directory and file name

data_dir = '../Receiver_Positions/'
data_file = 'rx_pts_depop_1078.csv'

results_dir = '../Generated_Data/'
results_file_base = 'Gen_Hollister_Test_c_5m_kx_ky_kz_'

n_cpu = 1
n_samp = 12
#********** INPUT PARAMS **********
ismag    = True
istensor = False

# ambient magnetic flux B[nT]

Bx = 4594.8
By = 19887.1 
Bz = 41568.2
# Bv = np.sqrt(Bx**2 + By**2 + Bz**2)

# measurement direction
LX = 1.
LY = 1.
LZ = 1.
"""


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
def main(args=None):
    """Console script for sciml_data_generator."""
    click.echo(
        "Replace this message by putting your code into "
        "sciml_data_generator.cli.main"
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

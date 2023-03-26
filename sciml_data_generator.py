"""Console script for sciml_data_generator."""
import sys
import click
import os
import numpy as np
import time
import yaml
import ctypes
import calc_and_mig_kx_ky_kz
import logging
from rich.logging import RichHandler
from tqdm import tqdm

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("rich")


def load_mesh_files(file_dir, base_file):
    # read the no. of nodes and cells
    node_file_name = file_dir + "/" + base_file + ".1.node"
    element_file_name = file_dir + "/" + base_file + ".1.ele"

    if os.path.exists(node_file_name):
        f = open(node_file_name, "r")
        line = f.readline()
        nnodes = int(line.split()[0])
        f.close()
        nodes = np.loadtxt(
            node_file_name, usecols=(1, 2, 3), skiprows=1, max_rows=nnodes, dtype=float
        )
    else:
        msg = "File %s doesn't exist or not accessible" % node_file_name
        log.error(msg)
        raise ValueError(msg)

    if os.path.exists(element_file_name):
        f = open(element_file_name, "r")
        line = f.readline()
        ncels = int(line.split()[0])
        f.close()
        eles = np.loadtxt(
            element_file_name,
            usecols=(1, 2, 3, 4),
            skiprows=1,
            max_rows=ncels,
            dtype=int,
        )
        eles = eles - 1
    else:
        msg = "File %s doesn't exist or not accessible" % element_file_name
        log.error(msg)
        raise ValueError(msg)

    return nnodes, ncels, nodes, eles


def load_rx_file(reciever_locations_file):
    rx_f = np.loadtxt(reciever_locations_file, delimiter=",")
    return rx_f.shape[0], rx_f


def load_mdl_npy_file(file_dir, base_file):
    mdl_val = np.load(file_dir + base_file + ".npy")
    return mdl_val


def centet(ndes, tetnds):
    nk = tetnds[:, 0]
    nl = tetnds[:, 1]
    nm = tetnds[:, 2]
    nn = tetnds[:, 3]
    return (ndes[nk, :] + ndes[nl, :] + ndes[nm, :] + ndes[nn, :]) / 4.0


def vol_tets(ndes, tetnds):
    ntt = len(tetnds)
    vot = np.zeros((ntt))
    for itet in np.arange(0, ntt):
        n1 = tetnds[itet, 0]
        n2 = tetnds[itet, 1]
        n3 = tetnds[itet, 2]
        n4 = tetnds[itet, 3]
        x1 = ndes[n1, 0]
        y1 = ndes[n1, 1]
        z1 = ndes[n1, 2]
        x2 = ndes[n2, 0]
        y2 = ndes[n2, 1]
        z2 = ndes[n2, 2]
        x3 = ndes[n3, 0]
        y3 = ndes[n3, 1]
        z3 = ndes[n3, 2]
        x4 = ndes[n4, 0]
        y4 = ndes[n4, 1]
        z4 = ndes[n4, 2]
        pv = (
            (x4 - x1) * ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
            + (y4 - y1) * ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
            + (z4 - z1) * ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        )
        vot[itet] = np.abs(pv / 6.0)
    return vot


def calc_and_migrate_field_one_mdl(
    x,
    ntets,
    rho_sus,
    Bv,
    KXt,
    KYt,
    KZt,
    LX,
    LY,
    LZ,
    nodes,
    tets,
    obs_pts,
    n_obs,
    ctet,
    vtet,
    mdl_dir,
    sus_name,
    kx_name,
    ky_name,
    kz_name,
    ismag,
    istensor,
):
    # global rho_sus, KXt, KYt, KZt
    sus_file = mdl_dir + sus_name + str(x) + ".npy"
    sus_val = np.load(sus_file)
    rho_sus[0:ntets] = sus_val

    kx_file = mdl_dir + kx_name + str(x) + ".npy"
    kx_val = np.load(kx_file)
    KXt[0:ntets] = kx_val

    ky_file = mdl_dir + ky_name + str(x) + ".npy"
    ky_val = np.load(ky_file)
    KYt[0:ntets] = ky_val

    kz_file = mdl_dir + kz_name + str(x) + ".npy"
    kz_val = np.load(kz_file)
    KZt[0:ntets] = kz_val

    if ismag:
        rho_sus = rho_sus * Bv

    # Load the shared object file
    # lib_path = 'sciml_data_generator/lib/calc_and_mig_kx_ky_kz.cpython-37m-darwin.so'
    # calc_and_mig_kx_ky_kz = ctypes.cdll.LoadLibrary(lib_path)

    # Call a function in the shared object file

    mig_data = calc_and_mig_kx_ky_kz.calc_and_mig_field(
        rho_sus,
        ismag,
        istensor,
        KXt,
        KYt,
        KZt,
        LX,
        LY,
        LZ,
        nodes,
        tets,
        ntets,
        obs_pts,
        n_obs,
        ctet,
        vtet,
    )
    return mig_data[0:ntets]


@click.command()
@click.option(
    "--config_file",
    type=click.Path(),
    help="Configuration file in YAML format",
    required=True,
    show_default=True,
)
def main(config_file):
    """SciML Data Generator"""
    if not os.path.exists(config_file):
        msg = "Configuration file %s doesn't exist or not accessible" % config_file
        log.error(msg)
        raise ValueError(msg)

    with open(config_file, "r") as fp:
        config = yaml.safe_load(fp)

    Bx = config["Bx"]
    By = config["By"]
    Bz = config["Bz"]

    Bv = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
    LX = np.float32(Bx / Bv)
    LY = np.float32(By / Bv)
    LZ = np.float32(Bz / Bv)

    mesh_dir = config["mesh"]["dir_name"]
    base_file_name = config["mesh"]["base_file_name"]

    if not os.path.exists(mesh_dir):
        msg = "Directory %s doesn't exist or not accessible" % mesh_dir
        log.error(msg)
        raise ValueError(msg)

    log.debug("Extracting nodes and elements from mesh")
    nverts, ntets, verts, tetra = load_mesh_files(mesh_dir, base_file_name)
    log.debug("Extraction done!")
    log.debug("nnodes: %d" % nverts)
    log.debug("ncells: %d" % ntets)

    reciever_locations_file = config["reciever_locations_file"]
    if not os.path.exists(reciever_locations_file):
        msg = "File %s doesn't exist or not accessible" % reciever_locations_file
        log.error(msg)
        raise ValueError(msg)

    # obs pt coors in m
    n_obs, rx_loc = load_rx_file(reciever_locations_file)
    log.debug("n_obs pts: %d" % n_obs)

    log.debug("Running Centet")
    ctt = centet(verts, tetra)
    log.debug("Running Centet done!")
    log.debug("Running Vol tets")
    vtt = vol_tets(verts, tetra)
    log.debug("Running Vol tets done!")

    # FORTRAN Input Param Initialization
    rho_sus = np.zeros((10000000), dtype="float32")
    KXt = np.zeros((10000000), dtype="float32")
    KYt = np.zeros((10000000), dtype="float32")
    KZt = np.zeros((10000000), dtype="float32")
    ctet = np.zeros((10000000, 3), dtype="float32")
    vtet = np.zeros((10000000), dtype="float32")

    nodes = np.zeros((10000000, 3), dtype="float32")
    tets = np.zeros((10000000, 4), dtype=int)
    obs_pts = np.zeros((1000000, 3), dtype="float32")

    nodes[0:nverts] = np.float32(verts)
    tets[0:ntets] = tetra + 1
    ctet[0:ntets] = np.float32(ctt)
    vtet[0:ntets] = np.float32(vtt)

    obs_pts[0:n_obs] = np.float32(rx_loc[:, 0:3])

    # mdl_id = [2]
    n_samp = config["n_samp"]
    mdl_id = [ix for ix in range(n_samp)]
    mig_field_all = np.zeros((len(mdl_id), ntets), dtype="float32")

    mdl_dir = config["model"]["dir_name"]
    sus_name = config["model"]["sus_name_prefix"]
    kx_name = config["model"]["kx_name_prefix"]
    ky_name = config["model"]["ky_name_prefix"]
    kz_name = config["model"]["kz_name_prefix"]
    ismag = config["ismag"]
    istensor = config["istensor"]

    mig_field_all = []
    for _id in tqdm(mdl_id):
        mig_field = calc_and_migrate_field_one_mdl(
            _id,
            ntets,
            rho_sus,
            Bv,
            KXt,
            KYt,
            KZt,
            LX,
            LY,
            LZ,
            nodes,
            tets,
            obs_pts,
            n_obs,
            ctet,
            vtet,
            mdl_dir,
            sus_name,
            kx_name,
            ky_name,
            kz_name,
            ismag,
            istensor,
        )
        mig_field_all.append(mig_field)

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover

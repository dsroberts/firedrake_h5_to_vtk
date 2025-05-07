import firedrake as fd
import sys
import argparse
import pickle
import os
import logging
from collections import defaultdict
from enum import Enum
from petsc4py import PETSc
from typing import Iterable

### Grab some constants from the firedrake checkpointing module
from firedrake.checkpointing import PREFIX, PREFIX_EMBEDDED


class Constants(str, Enum):
    prefix = PREFIX
    topologies = "topologies"
    global_mesh_map_attr = prefix + "_mesh_name_topology_name_map"
    function_space_attr = prefix + "_function_name_function_space_name_map"
    mixed_function_space_attr = prefix + "_mixed_function_name_mixed_function_space_name_map"
    meshes_group = prefix + "_meshes"
    mixed_meshes_group = prefix + "_mixed_meshes"
    mixed_function_space_group = prefix + "_mixed_function_spaces"
    functions_group = prefix + "_functions"
    functions_attr = prefix + "_function"
    firedrake_embedded = PREFIX_EMBEDDED

    def __str__(self):
        return self.value


class FunctionFilter:
    def __init__(self, func_list: str):
        if func_list == "all":
            self._filter: set[str] = set()
        else:
            self._filter = set(func for func in func_list.split(","))

    def __call__(self, in_fn: str) -> bool:
        ### Global filters
        if in_fn.startswith(Constants.firedrake_embedded):
            return False
        ### User input filters
        if self._filter:
            if in_fn in self._filter:
                return True
            return False
        else:
            return True

    def __iadd__(self, other: str | Iterable[str]):
        if self._filter:
            if isinstance(other, str):
                self._filter.add(other)
            else:
                self._filter |= set(other)
        return self


class NullHelpFormatter(argparse.HelpFormatter):
    def format_help(self):
        pass


def configure_petsc():
    ### Suppress PETSc 'unused options' warning
    opts = PETSc.Options()
    opts["options_left"] = 0


def configure_firedrake():
    pass


def configure_logger(verbose: bool, mpi_rank: int) -> logging.Logger:
    logger = logging.getLogger("h5_to_vtk")
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    if mpi_rank == 0:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt=logging.Formatter("%(levelname)s %(message)s"))
    else:
        handler = logging.NullHandler()
    logger.addHandler(handler)
    return logger


def main():
    configure_petsc()
    configure_firedrake()

    ### Make sure help text only appears on a single rank
    if fd.COMM_WORLD.rank == 0:
        arg_formatter = argparse.ArgumentDefaultsHelpFormatter
    else:
        arg_formatter = NullHelpFormatter

    parser = argparse.ArgumentParser(
        prog="firedrake_h5_to_vtk",
        description="Convert firedrake HDF5 checkpoint files to VTK output",
        formatter_class=arg_formatter,
    )
    parser.add_argument("-i", "--infile", required=True, help="The input firedrake checkpoint file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-o", "--outfile", help="Path to final output VTK file. Must end in '.pvd'")
    group.add_argument(
        "-l",
        "--list",
        help="List firedrake objects in HDF5 file. If a filter is supplied, results will be filtered",
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--functions",
        required=False,
        help="Comma separeted list of functions to convert. Must match internal function names discovered by firedrake_h5_to_vtk (i.e. output of firedrake_h5_to_vtk -l). Leave blank for all",
        default="all",
    )
    parser.add_argument("-v", "--verbose", help="Be noisy", action="store_true")

    ns = parser.parse_args(sys.argv[1:])
    logger = configure_logger(ns.verbose, fd.COMM_WORLD.rank)

    func_filter = FunctionFilter(ns.functions)

    ### Firedrake will tell us if the checkpoint file isn't a checkpoint file
    logger.info(f"Attempting to open {ns.infile}")
    checkpoint_file = fd.CheckpointFile(ns.infile, mode="r")

    ### Discover meshes
    logger.info(f"Loading global mesh map at {Constants.topologies}/{Constants.global_mesh_map_attr}")
    topo_dict = pickle.loads(checkpoint_file.h5pyfile[Constants.topologies].attrs[Constants.global_mesh_map_attr])

    ### Discover functions
    ### Function metadata can be in one of 2 places:
    ### cf['topologies/firedrake_mixed_meshes/{m}'].attrs[PREFIX + '_mixed_function_name_mixed_function_space_name_map'] for m in topo_dict.keys()
    ### cf['topologies/{m[1]}/firedrake_meshes/{m[0]}'].attrs[PREFIX + '_function_name_function_space_name_map'] for m in topo_dict.items()
    functions = defaultdict(dict)
    mixed_functions = defaultdict(dict)
    mixed_functions_components = defaultdict(set)
    for mesh_name, topology in topo_dict.items():
        group_path = os.path.join(Constants.topologies, Constants.mixed_meshes_group, mesh_name)
        logger.info(f"Checking for mixed functions on mesh {mesh_name} at path {group_path}")

        if group_path in checkpoint_file.h5pyfile:
            if Constants.mixed_function_space_attr in checkpoint_file.h5pyfile[group_path].attrs:
                funcs_from_h5 = pickle.loads(
                    checkpoint_file.h5pyfile[group_path].attrs[Constants.mixed_function_space_attr]
                )
                logger.info(f"   Found {[i for i in funcs_from_h5.keys()]}")
                mixed_functions[mesh_name].update(funcs_from_h5)
                ### Discover components of mixed function spaces
                for func_name, func_desc in funcs_from_h5.items():
                    mixed_func_component_path = os.path.join(
                        group_path,
                        Constants.mixed_function_space_group,
                        func_desc,
                        Constants.functions_group,
                        func_name,
                    )
                    logger.info(f"   Checking for components of {func_name} on {mixed_func_component_path}")
                    if mixed_func_component_path in checkpoint_file.h5pyfile:
                        for sub_func in checkpoint_file.h5pyfile[mixed_func_component_path]:
                            sub_func_group = os.path.join(mixed_func_component_path, sub_func)
                            if Constants.functions_attr in checkpoint_file.h5pyfile[sub_func_group].attrs:
                                mixed_functions_components[func_name].add(
                                    checkpoint_file.h5pyfile[sub_func_group].attrs[Constants.functions_attr]
                                )
                                logger.info(
                                    f"   Found {checkpoint_file.h5pyfile[sub_func_group].attrs[Constants.functions_attr]}"
                                )

        group_path = os.path.join(Constants.topologies, topology, Constants.meshes_group, mesh_name)
        logger.info(f"Checking for functions on mesh {mesh_name} at path {group_path}")

        if group_path in checkpoint_file.h5pyfile:
            if Constants.function_space_attr in checkpoint_file.h5pyfile[group_path].attrs:
                funcs_from_h5 = pickle.loads(checkpoint_file.h5pyfile[group_path].attrs[Constants.function_space_attr])
                functions[mesh_name].update(funcs_from_h5)
                logger.info(f"   Found {[i for i in funcs_from_h5.keys()]}")

    ### Filter
    for mesh_name in mixed_functions:
        logger.info(f"Filtering mixed functions on {mesh_name}")
        mixed_functions[mesh_name] = [func for func in mixed_functions[mesh_name] if func_filter(func)]
        logger.info(f"  Post-filter: {mixed_functions[mesh_name]}")

    ### If a mixed function has been requested, we have to ensure the subfunctions get written
    ### regardless of the filter
    for funcs in mixed_functions.values():
        for func in funcs:
            if func in mixed_functions_components:
                func_filter += mixed_functions_components[func]

    for mesh_name in functions:
        logger.info(f"Filtering functions on {mesh_name}")
        functions[mesh_name] = [func for func in functions[mesh_name] if func_filter(func)]
        logger.info(f"  Post-filter: {functions[mesh_name]}")

    if ns.list:
        if fd.COMM_WORLD.rank == 0:
            print("Discovered meshes:")
            for mesh_name in topo_dict.keys():
                if functions[mesh_name] or mixed_functions[mesh_name]:
                    print(mesh_name)

            print("\nDiscovered Functions:")
            for mesh_name in topo_dict.keys():
                for f in functions[mesh_name]:
                    print(f" {f} on mesh {mesh_name}")

            print("\nDiscovered Mixed Functions:")
            for mesh_name in topo_dict.keys():
                for f in mixed_functions[mesh_name]:
                    print(f" {f} on mesh {mesh_name}")
                    for sub_func in mixed_functions_components[f]:
                        print(f"  {f} has subfunction {sub_func}")
        return

    if not ns.outfile.endswith(".pvd"):
        logger.error("Output file must end in '.pvd'")
        return

    ### Load
    meshes_to_load = [mesh_name for mesh_name in topo_dict.keys() if functions[mesh_name] or mixed_functions[mesh_name]]
    ### Warn if we have more than one mesh to load
    if len(meshes_to_load) > 1:
        logger.warning(
            """Multiple meshes requested. PETSc usually crashes when attempting to load multiple meshes in the 
            same session. Use the -f option to limit output to functions on a single mesh.
            """
        )
    if len(meshes_to_load) == 0:
        logger.warning("No valid functions selected")
        return

    functions_to_write = []
    for mesh_name in meshes_to_load:
        logger.info(f"Loading mesh {mesh_name}")
        mesh = checkpoint_file.load_mesh(mesh_name)
        for func in functions[mesh_name]:
            logger.info(f"   Loading function {func}")
            functions_to_write.append(checkpoint_file.load_function(mesh, func))
        logger.info("Writing VTK output")
        fd.VTKFile(ns.outfile).write(*functions_to_write)
        del mesh


if __name__ == "__main__":
    main()

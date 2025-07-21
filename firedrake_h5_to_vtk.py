import firedrake as fd
import numpy as np
import numpy.typing as npt
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
    dms = "dms"
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


class TimestepFilter:
    def __init__(self, ts_list: str):
        if ts_list == "all":
            self.filter = "all"
        elif ts_list == "last":
            self.filter = "last"
        else:
            self.filter = list(int(i) for i in ts_list.split((",")))

    def __call__(self, timesteps: npt.NDArray[np.float64]) -> list[int]:
        if self.filter == "all":
            return [int(i) for i in timesteps]
        elif self.filter == "last":
            return [int(timesteps[-1])]
        else:
            return [int(i) for i in timesteps if int(i) in self.filter]


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
        help="Comma separeted list of functions to convert. Must match internal function names discovered by firedrake_h5_to_vtk (i.e. output of firedrake_h5_to_vtk -l). Leave blank for all. Ignored when the '-l' option is passed",
        default="all",
    )
    parser.add_argument(
        "-t",
        "--timesteps",
        required=False,
        help="Comma separated list of timesteps to retrieve from the file. Special values are 'all' for all timesteps and 'last' for the final timestep. Leave blank for all. Ignored when no timestepping is detected in the data",
        default="all",
    )
    parser.add_argument("-v", "--verbose", help="Be noisy", action="store_true")

    ns = parser.parse_args(sys.argv[1:])
    logger = configure_logger(ns.verbose, fd.COMM_WORLD.rank)

    ts_filter = TimestepFilter(ns.timesteps)
    if ns.list:
        func_filter = FunctionFilter("all")
    else:
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
    timestepping = defaultdict(dict)
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

        group_path = os.path.join(Constants.topologies, topology, Constants.dms)
        logger.info(f"Checking for timestepping data on mesh {mesh_name} on path {group_path}")
        ### Mixed functions do not have timestepping data
        ### If a firedrake_embedded_<function> group exists, the timestepping will be there and not
        ### in <function>
        ### Timestepping info will be duplicated between the firedrake_embedded_<function> and <function>
        ### paths as firedrake_embedded_<function>s are detected separately, however the duplicates will be
        ### dropped at the filtering stage
        for func in functions[mesh_name]:
            logger.info(f"  Checking for {func}")
            for gpath in checkpoint_file.h5pyfile[group_path]:
                full_gpath = os.path.join(
                    group_path, gpath, "vecs", "firedrake_embedded_" + func, "firedrake_timestepping"
                )
                logger.info(f"    Checking at {full_gpath}")
                if full_gpath in checkpoint_file.h5pyfile:
                    if "firedrake_timestepping_history_index" in checkpoint_file.h5pyfile[full_gpath].attrs:
                        timestepping[mesh_name][func] = [
                            int(i)
                            for i in checkpoint_file.h5pyfile[full_gpath].attrs["firedrake_timestepping_history_index"]
                        ]
                        logger.info(f"    Found embedded timestepping data at {full_gpath}")
                        break
                full_gpath = os.path.join(group_path, gpath, "vecs", func, "firedrake_timestepping")
                logger.info(f"    Checking at {full_gpath}")
                if full_gpath in checkpoint_file.h5pyfile:
                    if "firedrake_timestepping_history_index" in checkpoint_file.h5pyfile[full_gpath].attrs:
                        timestepping[mesh_name][func] = [
                            int(i)
                            for i in checkpoint_file.h5pyfile[full_gpath].attrs["firedrake_timestepping_history_index"]
                        ]
                        logger.info(f"    Found timestepping data at {full_gpath}")
                        break

    ### Filter
    for mesh_name in mixed_functions:
        logger.info(f"Filtering mixed functions on {mesh_name}")
        mixed_functions[mesh_name] = {func: v for func, v in mixed_functions[mesh_name].items() if func_filter(func)}
        logger.info(f"  Post-filter: {mixed_functions[mesh_name]}")

    ### If a mixed function has been requested, we have to ensure the subfunctions get written
    ### regardless of the filter
    for funcs in mixed_functions.values():
        for func in funcs:
            if func in mixed_functions_components:
                func_filter += mixed_functions_components[func]

    for mesh_name, topology in topo_dict.items():
        logger.info(f"Filtering functions on {mesh_name}")
        functions[mesh_name] = {func: v for func, v in functions[mesh_name].items() if func_filter(func)}
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
                    if f in timestepping[mesh_name]:
                        print(f"  {f} has timesteps {', '.join([str(i) for i in timestepping[mesh_name][f]])}")

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
    output_file = fd.VTKFile(ns.outfile)
    timestepping_details = {}
    has_timestepping = False
    for mesh_name in meshes_to_load:
        logger.info(f"Loading mesh {mesh_name}")
        mesh = checkpoint_file.load_mesh(mesh_name)
        for func in functions[mesh_name]:
            logger.info(f"   Getting timestepping information for {func}")
            tsh = checkpoint_file.get_timestepping_history(mesh, func)
            if tsh:
                has_timestepping = True
                logger.info("   Found timestepping information:")
                timestepping_details[func] = ts_filter(tsh["index"])
                logger.info(f"     {func} timesteps to write:")
                logger.info("       " + ", ".join(str(i) for i in timestepping_details[func]))
            else:
                logger.info("   No timestepping information")

        all_timesteps = sorted(set(j for i in timestepping_details.values() for j in i))
        if all_timesteps:
            for ts in all_timesteps:
                functions_to_write = []
                for func in functions[mesh_name]:
                    if ts in timestepping_details[func]:
                        functions_to_write.append(checkpoint_file.load_function(mesh, func, idx=ts))
                if functions_to_write:
                    logger.info(f"Writing VTK output for timestep {ts}")
                    output_file.write(*functions_to_write)
        else:
            if has_timestepping:
                logger.error("No valid timesteps selected!")
                exit(1)
            for func in functions[mesh_name]:
                functions_to_write.append(checkpoint_file.load_function(mesh, func))
            logger.info("Writing VTK output")
            output_file.write(*functions_to_write)
        del mesh


if __name__ == "__main__":
    main()

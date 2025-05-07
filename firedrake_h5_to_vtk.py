import firedrake as fd
import sys
import argparse
import pickle
import os
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


def configure_petsc():
    ### Suppress PETSc 'unused options' warning
    opts = PETSc.Options()
    opts["options_left"] = 0


def configure_firedrake():
    pass


def main():
    configure_petsc()
    configure_firedrake()

    parser = argparse.ArgumentParser(
        prog="firedrake_h5_to_vtk", description="Convert firedrake HDF5 checkpoint files to VTK output"
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
        "--fields",
        required=False,
        help="Comma separeted list of fields to convert. Must match internal G-ADOPT field names. Leave blank for all",
        default="all",
    )
    parser.add_argument("-v", "--verbose", help="Be noisy", action="store_true")

    ns = parser.parse_args(sys.argv[1:])
    func_filter = FunctionFilter(ns.fields)

    ### Firedrake will tell us if the checkpoint file isn't a checkpoint file
    if ns.verbose:
        print("Attempting to open " + ns.infile)
    checkpoint_file = fd.CheckpointFile(ns.infile, mode="r")

    ### Discover meshes
    if ns.verbose:
        print(f"Loading global mesh map at {Constants.topologies}/{Constants.global_mesh_map_attr}")
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
        if ns.verbose:
            print(f"Checking for mixed functions on mesh {mesh_name} at path {group_path}")

        if group_path in checkpoint_file.h5pyfile:
            if Constants.mixed_function_space_attr in checkpoint_file.h5pyfile[group_path].attrs:
                funcs_from_h5 = pickle.loads(
                    checkpoint_file.h5pyfile[group_path].attrs[Constants.mixed_function_space_attr]
                )
                if ns.verbose:
                    print(f"   Found {[i for i in funcs_from_h5.keys()]}")
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
                    if ns.verbose:
                        print(f"   Checking for components of {func_name} on {mixed_func_component_path}")
                    if mixed_func_component_path in checkpoint_file.h5pyfile:
                        for sub_func in checkpoint_file.h5pyfile[mixed_func_component_path]:
                            sub_func_group = os.path.join(mixed_func_component_path, sub_func)
                            if Constants.functions_attr in checkpoint_file.h5pyfile[sub_func_group].attrs:
                                mixed_functions_components[func_name].add(
                                    checkpoint_file.h5pyfile[sub_func_group].attrs[Constants.functions_attr]
                                )
                                if ns.verbose:
                                    print(
                                        f"   Found {checkpoint_file.h5pyfile[sub_func_group].attrs[Constants.functions_attr]}"
                                    )

        group_path = os.path.join(Constants.topologies, topology, Constants.meshes_group, mesh_name)
        if ns.verbose:
            print(f"Checking for functions on mesh {mesh_name} at path {group_path}")

        if group_path in checkpoint_file.h5pyfile:
            if Constants.function_space_attr in checkpoint_file.h5pyfile[group_path].attrs:
                funcs_from_h5 = pickle.loads(checkpoint_file.h5pyfile[group_path].attrs[Constants.function_space_attr])
                functions[mesh_name].update(funcs_from_h5)
                if ns.verbose:
                    print(f"   Found {[i for i in funcs_from_h5.keys()]}")

    ### Filter
    for mesh_name in mixed_functions:
        if ns.verbose:
            print(f"Filtering mixed functions on {mesh_name}")
        mixed_functions[mesh_name] = [func for func in mixed_functions[mesh_name] if func_filter(func)]
        if ns.verbose:
            print(f"  Post-filter: {mixed_functions[mesh_name]}")

    ### If a mixed function has been requested, we have to ensure the subfunctions get written
    ### regardless of the filter
    for funcs in mixed_functions.values():
        for func in funcs:
            if func in mixed_functions_components:
                func_filter += mixed_functions_components[func]

    for mesh_name in functions:
        if ns.verbose:
            print(f"Filtering functions on {mesh_name}")
        functions[mesh_name] = [func for func in functions[mesh_name] if func_filter(func)]
        if ns.verbose:
            print(f"  Post-filter: {functions[mesh_name]}")

    if ns.list:
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
        print("ERROR: Output file must end in '.pvd'",file=sys.stderr)
        return

    ### Load
    meshes_to_load = [mesh_name for mesh_name in topo_dict.keys() if functions[mesh_name] or mixed_functions[mesh_name]]
    ### Warn if we have more than one mesh to load
    if len(meshes_to_load) > 1:
        print(
            """WARNING: multiple meshes. PETSc usually crashes when attempting to load multiple meshes in the 
            same session. Use the -f option to limit output to functions on a single mesh.
            """,
            file=sys.stderr,
        )
    if len(meshes_to_load) == 0:
        print("No valid fields selected", file=sys.stderr)
        return

    functions_to_write = []
    for mesh_name in meshes_to_load:
        if ns.verbose:
            print(f"Loading mesh {mesh_name}")
        mesh = checkpoint_file.load_mesh(mesh_name)
        for func in functions[mesh_name]:
            if ns.verbose:
                print(f"   Loading function {func}")
            functions_to_write.append(checkpoint_file.load_function(mesh, func))
        if ns.verbose:
            print("Writing VTK output")
        fd.VTKFile(ns.outfile).write(*functions_to_write)
        del mesh


if __name__ == "__main__":
    main()

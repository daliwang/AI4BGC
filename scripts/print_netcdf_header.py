#!/usr/bin/env python3
import argparse
from pathlib import Path
from netCDF4 import Dataset


def format_dim_len(dim) -> str:
    if dim.isunlimited():
        # In netCDF, unlimited dims shown as UNLIMITED ; currently N
        return f"UNLIMITED ; ({len(dim)} currently)"
    return str(len(dim))


def print_header(nc_path: str, show_var_attrs: bool = True, show_global_attrs: bool = True) -> None:
    nc_path = str(nc_path)
    with Dataset(nc_path, 'r') as ds:
        print(f"netcdf {Path(nc_path).name} {{")

        # Dimensions
        print("  dimensions:")
        for name, dim in ds.dimensions.items():
            print(f"    {name} = {format_dim_len(dim)} ;")

        # Variables
        print("\n  variables:")
        for name, var in ds.variables.items():
            dims = ", ".join(var.dimensions)
            print(f"    {var.dtype} {name}({dims}) ;")
            if show_var_attrs:
                for attr in var.ncattrs():
                    val = getattr(var, attr)
                    # Render strings with quotes, lists compactly
                    if isinstance(val, str):
                        sval = f'"{val}"'
                    else:
                        sval = f"{val}"
                    print(f"      {name}:{attr} = {sval} ;")

        # Global attributes
        if show_global_attrs:
            print("\n  // global attributes:")
            for attr in ds.ncattrs():
                val = getattr(ds, attr)
                if isinstance(val, str):
                    sval = f'"{val}"'
                else:
                    sval = f"{val}"
                print(f"    :{attr} = {sval} ;")

        print("}")


def main():
    p = argparse.ArgumentParser(description="Print a NetCDF header (like ncdump -h)")
    p.add_argument("nc", help="Path to NetCDF file (.nc)")
    p.add_argument("--no-var-attrs", action="store_true", help="Do not print per-variable attributes")
    p.add_argument("--no-global-attrs", action="store_true", help="Do not print global attributes")
    args = p.parse_args()

    print_header(
        args.nc,
        show_var_attrs=not args.no_var_attrs,
        show_global_attrs=not args.no_global_attrs,
    )


if __name__ == "__main__":
    main() 
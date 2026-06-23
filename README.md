# OceanMed.jl

High-resolution Mediterranean Sea ocean simulations built on
[Oceananigans](https://github.com/CliMA/Oceananigans.jl) and
[NumericalEarth](https://github.com/CliMA/NumericalEarth.jl).

`OceanMed` is both a small reusable module (`src/`) and a set of runnable example scripts. The module
registers the high-resolution Copernicus Mediterranean bathymetry as a NumericalEarth dataset and
collects the building blocks shared across the Mediterranean configurations — vertical-grid
construction, the Strait of Gibraltar open boundary conditions, and the sponge layers.

![Mediterranean bottom height regridded from the Copernicus 4.2 km bathymetry](docs/assets/mediterranean_bathymetry.png)

## What is in here

| File | Purpose |
| --- | --- |
| `src/OceanMed.jl` | Module entry point and exports. |
| `src/mediterranean_bathymetry.jl` | `MEDSEABathymetry` — the Copernicus Mediterranean bathymetry registered as a NumericalEarth dataset. |
| `src/vertical_grids.jl` | `copernicus_z_faces` — the stretched Copernicus vertical grid. |
| `src/open_boundary_conditions.jl` | Gibraltar open boundary conditions and sponge forcings. |
| `mediterranean_simulation.jl` | The main Mediterranean example (serial / GPU). |
| `distributed_mediterranean_simulation.jl` | The distributed (MPI) Mediterranean example. |
| `adriatic_simulation.jl` | An Adriatic configuration using its own mesh-mask bathymetry. |
| `download_glorys_med.jl`, `download_files.jl` | Helpers to pre-download forcing/boundary data on a login node. |

## Requirements

- Julia ≥ 1.10.
- A [Copernicus Marine](https://data.marine.copernicus.eu/register) account (free) for the bathymetry
  and the GLORYS boundary data, exported as environment variables:

  ```bash
  export COPERNICUS_USERNAME="your-username"
  export COPERNICUS_PASSWORD="your-password"
  ```

  The `copernicusmarine` command-line tool is installed automatically through `CopernicusMarine.jl`
  (via CondaPkg) on first use.

> **Note on Oceananigans.** The open boundary conditions used here (the `Radiation` and `Flather`
> schemes) live on the `ss/open-boundary-conditions` branch of Oceananigans. `Manifest.toml` already
> pins it, so `Pkg.instantiate()` fetches the right revision — no manual action needed.

## Setup

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## The high-resolution Mediterranean bathymetry

`MEDSEABathymetry` registers the static bottom topography (`deptho`) of the Copernicus product
[`MEDSEA_ANALYSISFORECAST_PHY_006_013`](https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/description)
(≈4.2 km, 1/24°) as a NumericalEarth dataset, so it plugs straight into `regrid_bathymetry`. The
download (dataset `cmems_mod_med_phy_anfc_4.2km_static`, part `bathy`) is normalized from a positive
`deptho` to a signed bottom height before regridding.

```julia
using OceanMed, NumericalEarth, Oceananigans

grid = LatitudeLongitudeGrid(size = (1350, 540, 1),
                             longitude = (-6, 36.5), latitude = (30, 46),
                             z = (-5000, 0), halo = (7, 7, 1))

bathymetry = Metadatum(:bottom_height; dataset = MEDSEABathymetry(), dir = "./data")
bottom_height = regrid_bathymetry(grid, bathymetry; minimum_depth = 5, major_basins = 1)
```

Swap `MEDSEABathymetry()` for `ETOPO2022()` to fall back to the global default bathymetry.

## Running the Mediterranean simulation

```bash
julia --project=. mediterranean_simulation.jl
```

The script builds a ~1/30° Mediterranean grid with the 280-level Copernicus vertical grid and the
high-resolution bathymetry, then runs a coupled ocean simulation forced by JRA55-do, with the only
open edge — the Strait of Gibraltar on the west — handled by GLORYS-fed open boundary conditions and
a sponge layer. It is set up for `GPU()`; switch `arch = CPU()` for a (much slower) CPU run.

The boundary data can be pre-downloaded on a login node with `download_glorys_med.jl`.

## Module API

- `copernicus_z_faces(; filepath, refinement)` — vertical face positions of the Copernicus
  Mediterranean grid, refined by an integer factor (default 2 → 280 levels).
- `MEDSEABathymetry` — the Copernicus Mediterranean bathymetry dataset (see above).
- `gibraltar_boundary_conditions(grid, u, v, T, S, η; inflow_timescale, outflow_timescale)` — the
  western open boundary: baroclinic `Radiation` conditions on `u, v, T, S` and a barotropic `Flather`
  condition on the transport `U`, all fed by GLORYS `FieldTimeSeries`. Use with a
  `SplitExplicitFreeSurface`.
- `gibraltar_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta; west_longitude, sponge_width, ...)`
  — `DatasetRestoring` forcings that relax the prognostic fields towards GLORYS inside a Gaussian
  sponge just behind the western boundary.
- `WesternSpongeMask(west_longitude, width)` — the Gaussian sponge mask used above.

Every exported symbol carries a docstring; use `?MEDSEABathymetry` (etc.) at the REPL for details.

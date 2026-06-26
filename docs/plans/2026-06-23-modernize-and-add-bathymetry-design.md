# OceanMed.jl modernization + Mediterranean bathymetry — design

Date: 2026-06-23
Branch: `ss/modernize-and-add-bathymetry`

## Goal

Modernize the OceanMed Mediterranean configuration so that it (1) tracks a current NumericalEarth, (2)
exposes its reusable pieces through an `OceanMed` module under `src/`, (3) registers a higher-resolution
Mediterranean bathymetry as a NumericalEarth `Metadata`-backed dataset downloaded from Copernicus Marine,
and (4) ships lightweight documentation.

The new bathymetry is the static bottom-depth field of the Copernicus product
`MEDSEA_ANALYSISFORECAST_PHY_006_013`, accessed through "Data Access" as the per-product email from
Emanuela indicates.

Scope is the Mediterranean: `mediterranean_simulation.jl` and its distributed variant. The Adriatic
script keeps its own mesh-mask bathymetry and is left untouched.

## Decisions already taken

- **NumericalEarth 0.2.3 → 0.5.6** (latest registered). The blocker for custom bathymetry — `regrid_bathymetry`
  hardcoding the NetCDF variable `"z"` — was fixed upstream in 0.5.3+ (now reads `dataset_variable_name(metadata)`).
  No fork, no dependency on a WIP dev branch.
- **Module = reusable pieces.** `src/` holds the new bathymetry dataset, vertical-grid construction, and the
  sponge/OBC helpers. The top-level scripts stay runnable but become thin drivers calling into `OceanMed`.
- **Copernicus identifiers (verified against the catalogue):**
  - `dataset_id = "cmems_mod_med_phy_anfc_4.2km_static"`
  - `dataset_part = "bathy"`
  - variable `deptho` (positive sea-floor depth in metres; `_FillValue` over land)

## Components

### 1. NumericalEarth 0.5.6 bump

Set `[compat] NumericalEarth = "0.5"` in `Project.toml`, resolve, precompile. Audit and repair the
0.2.3 → 0.5.x API delta where the Mediterranean script touches it (`regrid_bathymetry`, `Metadata`/`Metadatum`,
`ocean_simulation`, `OceanSeaIceModel`, `JRA55PrescribedAtmosphere`, GLORYS datasets, `set!`, `FieldTimeSeries`).
`regrid_bathymetry`'s `major_basins` keyword and the GLORYS interface (`GLORYSStatic`, `:depth => "deptho"`,
`copernicusmarine_dataset_id`) are stable across the bump.

### 2. `src/OceanMed.jl` module

```
src/OceanMed.jl                  # module entry: includes + a single export block
src/mediterranean_bathymetry.jl  # the new dataset registration (core deliverable)
src/vertical_grids.jl            # copernicus_z_faces(...) — the 280-level z, today inline in scripts
src/open_boundary_conditions.jl  # sponge forcings, west_boundary_value, drag-BC assembly (today duplicated)
```

The module mirrors NumericalEarth's own convention: include submodule files, then one top-level `export`.

### 3. Mediterranean bathymetry dataset — the core

`struct MEDSEABathymetry end`, conforming to the NumericalEarth dataset interface, following the ETOPO and
`GLORYSStatic` patterns:

- `copernicusmarine_dataset_id(::MEDSEABathymetry) = "cmems_mod_med_phy_anfc_4.2km_static"`.
- `download_dataset(::Metadatum{<:MEDSEABathymetry})`:
  1. `CopernicusMarine.copernicusmarine.subset(; dataset_id, dataset_part="bathy", variables=["deptho"], bbox, output…)`
     (credentials from `COPERNICUS_USERNAME`/`COPERNICUS_PASSWORD`, exactly as the GLORYS extension does).
  2. Normalize: read `deptho`, write a `bottom_height`/`z` variable equal to `-deptho`, with land
     (`_FillValue`/`missing`) set to a small positive elevation so `regrid_bathymetry`'s `z_data .> 0`
     land test fires. Save at `metadata_path`. Idempotent (skip if the normalized file exists).
- `dataset_variable_name(::…) = "z"` (the normalized variable).
- Geometry read **from the downloaded file** — `longitude_interfaces`, `latitude_interfaces`, `Base.size` —
  following the precedent of GLORYS `z_interfaces`, so the EAS grid is never hardcoded.
- Static dates `[nothing]`; `default_download_directory`.
- Region carried as a `BoundingBox` so the native grid gets a **Bounded** (regional) x-topology rather than
  the global `Periodic` default.

Result: `regrid_bathymetry(grid, Metadatum(:bottom_height; dataset=MEDSEABathymetry(), dir, bounding_box))`
works with stock NumericalEarth 0.5.6, at the Med model's native ~4.2 km resolution.

### 4. Script modernization (Mediterranean only)

- Inline z-face block → `OceanMed.copernicus_z_faces()`.
- ETOPO `regrid_bathymetry(grid; …)` → MED dataset for higher resolution (ETOPO stays available as a fallback).
- Inline sponge/BC blocks → module helpers.

### 5. Documentation

Docstrings on every public `OceanMed` symbol, plus a `docs/` Documenter build with: an index, a
"Mediterranean bathymetry" how-to (register + download + regrid), and the modernized Med example rendered via
Literate — mirroring NumericalEarth's own lightweight docs. Final heaviness TBD with Simone.

## Verification (CPU, no GPU needed)

1. Download a small Med sub-box through the registered dataset (credentials present).
2. Run `regrid_bathymetry` onto a coarse CPU `LatitudeLongitudeGrid`; assert the field is negative in the
   sea, land masked, finite — render a quick heatmap PNG.
3. Precompile the `OceanMed` module and load the modernized script up to model construction where feasible.

## Risks

- 0.2.3 → 0.5.6 API churn in the scripts (the full GPU run can't be exercised here; covered by precompile +
  CPU bathymetry/grid construction).
- `deptho` fill/mask handling and native-grid topology — resolved empirically against the real file.

module MediterraneanBathymetry

export MEDSEABathymetry

using NCDatasets
using CopernicusMarine
using Downloads: Downloads
using Oceananigans.DistributedComputations: @root

using NumericalEarth.DataWrangling: DataWrangling, AbstractStaticBathymetry,
                                    Metadata, Metadatum, metadata_path

"""
    MEDSEABathymetry

High-resolution (~1/24°, ≈4.2 km) Mediterranean Sea bottom topography, registered as a
NumericalEarth dataset so that it plugs directly into `regrid_bathymetry`.

The field is the static `deptho` (sea-floor depth) of the Copernicus Marine product
[`MEDSEA_ANALYSISFORECAST_PHY_006_013`](https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/description),
downloaded through the Copernicus Marine "Data Access" service (dataset
`cmems_mod_med_phy_anfc_4.2km_static`, part `bathy`). Downloading requires Copernicus
credentials in the `COPERNICUS_USERNAME` and `COPERNICUS_PASSWORD` environment variables
(free registration at https://data.marine.copernicus.eu/register).

`deptho` stores a positive depth that is `missing` over land, whereas `regrid_bathymetry`
expects a signed bottom height (negative below sea level, positive over land). The download
therefore normalizes the raw field to a `bottom_height = -deptho` variable named `z`, with
land set to a small positive elevation.

Example
=======

```julia
using OceanMed, NumericalEarth, Oceananigans

dataset = MEDSEABathymetry()
metadatum = Metadatum(:bottom_height; dataset, dir = "./data")

grid = LatitudeLongitudeGrid(size = (1350, 540, 1),
                             longitude = (-6, 36.5), latitude = (30, 46),
                             z = (-5000, 0), halo = (7, 7, 1))

bottom_height = regrid_bathymetry(grid, metadatum; minimum_depth = 5, major_basins = 1)
```
"""
struct MEDSEABathymetry <: AbstractStaticBathymetry end

const MEDSEAMetadata{D} = Metadata{<:MEDSEABathymetry, D}
const MEDSEAMetadatum   = Metadatum{<:MEDSEABathymetry}

const copernicus_dataset_id   = "cmems_mod_med_phy_anfc_4.2km_static"
const copernicus_dataset_part = "bathy"
const copernicus_variable     = "deptho"

# Elevation assigned to land cells in the normalized file. Positive so that
# `regrid_bathymetry`'s `z_data .> 0` land test fires on dry points.
const land_height = 10.0

const MEDSEA_variable_names = Dict(:bottom_height => "z")

DataWrangling.dataset_variable_name(metadata::MEDSEAMetadata) = MEDSEA_variable_names[metadata.name]
DataWrangling.default_download_directory(::MEDSEABathymetry) = pwd()
DataWrangling.reversed_vertical_axis(::MEDSEABathymetry) = false
DataWrangling.metadata_filename(::MEDSEABathymetry, name, date, region) = "medsea_bottom_height.nc"

raw_filename(::MEDSEABathymetry) = "medsea_deptho_raw.nc"

#####
##### Grid geometry read from the normalized file
#####
##### The Mediterranean model grid is a regular latitude–longitude mesh, so the native grid
##### is fully described by the outer face extents and the cell counts, both read from the
##### downloaded file (mirroring how GLORYS reads its `z_interfaces` from disk).
#####

function outer_face_extent(centers)
    spacing = centers[2] - centers[1]
    return (first(centers) - spacing / 2, last(centers) + spacing / 2)
end

function read_horizontal_coordinates(metadata::MEDSEAMetadata)
    path = metadata_path(metadata)
    isfile(path) || throw(ArgumentError("$(path) not found — download the dataset first."))
    Dataset(path) do dataset
        longitude = Float64.(dataset["longitude"][:])
        latitude  = Float64.(dataset["latitude"][:])
        return longitude, latitude
    end
end

DataWrangling.longitude_interfaces(metadata::MEDSEAMetadata) = outer_face_extent(first(read_horizontal_coordinates(metadata)))
DataWrangling.latitude_interfaces(metadata::MEDSEAMetadata)  = outer_face_extent(last(read_horizontal_coordinates(metadata)))

function Base.size(metadata::MEDSEAMetadata)
    longitude, latitude = read_horizontal_coordinates(metadata)
    return (length(longitude), length(latitude), 1, 1)
end

#####
##### Download and normalization
#####

function copernicus_credentials()
    username = get(ENV, "COPERNICUS_USERNAME", nothing)
    password = get(ENV, "COPERNICUS_PASSWORD", nothing)
    if isnothing(username) || isnothing(password)
        @warn "No Copernicus credentials found. Set the COPERNICUS_USERNAME and \
               COPERNICUS_PASSWORD environment variables to download the Mediterranean \
               bathymetry. Free registration: https://data.marine.copernicus.eu/register."
    end
    return username, password
end

# Reorder the raw `deptho` array to (longitude, latitude) regardless of the on-disk dimension order.
function as_longitude_latitude(variable)
    dims = NCDatasets.dimnames(variable)
    data = variable[:, :]
    return first(dims) == "longitude" ? data : permutedims(data, (2, 1))
end

function normalize_bathymetry(raw_path, output_path)
    Dataset(raw_path) do raw
        longitude = Float64.(raw["longitude"][:])
        latitude  = Float64.(raw["latitude"][:])
        deptho    = as_longitude_latitude(raw[copernicus_variable])

        bottom_height = map(deptho) do depth
            ismissing(depth) ? land_height : -Float64(depth)
        end

        Dataset(output_path, "c") do out
            defDim(out, "longitude", length(longitude))
            defDim(out, "latitude",  length(latitude))
            defVar(out, "longitude", longitude, ("longitude",))
            defVar(out, "latitude",  latitude,  ("latitude",))
            defVar(out, "z", bottom_height, ("longitude", "latitude"))
        end
    end
    return output_path
end

"""
    Downloads.download(metadatum::Metadatum{<:MEDSEABathymetry})

Download the Copernicus Marine Mediterranean static bathymetry and write the normalized
signed bottom-height file at `metadata_path(metadatum)`. Idempotent: returns immediately if
the normalized file already exists.
"""
function Downloads.download(metadatum::MEDSEAMetadatum)
    output_path = metadata_path(metadatum)
    isfile(output_path) && return output_path

    username, password = copernicus_credentials()
    raw_path = joinpath(metadatum.dir, raw_filename(metadatum.dataset))

    @root begin
        if !isfile(raw_path)
            @info "Downloading Mediterranean bathymetry ($(copernicus_dataset_id)/$(copernicus_dataset_part)) to $(metadatum.dir)..."
            toolbox = CopernicusMarine.copernicusmarine
            credential_kw = isnothing(username) || isnothing(password) ? NamedTuple() : (; username, password)
            toolbox.subset(; dataset_id = copernicus_dataset_id,
                             dataset_part = copernicus_dataset_part,
                             variables = CopernicusMarine.pylist([copernicus_variable]),
                             output_directory = metadatum.dir,
                             output_filename = raw_filename(metadatum.dataset),
                             coordinates_selection_method = "outside",
                             skip_existing = true,
                             credential_kw...)
        end
        normalize_bathymetry(raw_path, output_path)
    end

    return output_path
end

end # module MediterraneanBathymetry

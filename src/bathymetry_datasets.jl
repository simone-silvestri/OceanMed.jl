module BathymetryDatasets

export MEDSEABathymetry, EMODnetBathymetry

using NCDatasets
using CopernicusMarine
using Downloads: Downloads
using Oceananigans.DistributedComputations: @root

using NumericalEarth.DataWrangling: DataWrangling, AbstractStaticBathymetry,
                                    Metadata, Metadatum, metadata_path

#####
##### Two Mediterranean bathymetry datasets, normalized to a common signed-bottom-height file
#####
##### Both download a raw field, convert it to a `bottom_height` (negative below sea level, positive over
##### land) stored as variable `z`, and read their native lat–lon grid back from that normalized file.
##### `regrid_bathymetry` then consumes either one through the generic `AbstractStaticBathymetry` path.
#####

"""
    MEDSEABathymetry

High-resolution (~1/24°, ≈4.2 km) Mediterranean bathymetry — the static `deptho` of the Copernicus
Marine product `MEDSEA_ANALYSISFORECAST_PHY_006_013` (dataset `cmems_mod_med_phy_anfc_4.2km_static`,
part `bathy`). This is the model bathymetry at the model resolution. Requires Copernicus credentials in
`COPERNICUS_USERNAME` / `COPERNICUS_PASSWORD`.
"""
struct MEDSEABathymetry <: AbstractStaticBathymetry end

"""
    EMODnetBathymetry

Very-high-resolution (~1/16 arc-minute, ≈115 m) European bathymetry — the EMODnet Bathymetry DTM 2024,
downloaded open-access (no credentials) from the EMODnet ERDDAP server, subset to the metadata's
`region` bounding box. Use it to resolve coastal topography finer than the MEDSEA model bathymetry.

Pass a bounding box, e.g.

```julia
region = BoundingBox(longitude = (-6, 37), latitude = (30, 46))
metadatum = Metadatum(:bottom_height; dataset = EMODnetBathymetry(), dir = "./data", region)
```

The full-resolution European grid is large (the whole Mediterranean at 115 m is several GB); subset to
the region you actually need.
"""
struct EMODnetBathymetry <: AbstractStaticBathymetry end

const MEDSEAMetadata{D}  = Metadata{<:MEDSEABathymetry, D}
const EMODnetMetadata{D} = Metadata{<:EMODnetBathymetry, D}
const MEDSEAMetadatum    = Metadatum{<:MEDSEABathymetry}
const EMODnetMetadatum   = Metadatum{<:EMODnetBathymetry}

const NormalizedBathymetry = Union{MEDSEABathymetry, EMODnetBathymetry}
const NormalizedMetadata   = Union{MEDSEAMetadata, EMODnetMetadata}

# Elevation assigned to land / no-data cells in the normalized file, positive so that
# `regrid_bathymetry`'s `z_data .> 0` land test fires on dry points.
const land_height = 10.0

const bathymetry_variable_names = Dict(:bottom_height => "z")

DataWrangling.dataset_variable_name(metadata::NormalizedMetadata) = bathymetry_variable_names[metadata.name]
DataWrangling.reversed_vertical_axis(::NormalizedBathymetry) = false
DataWrangling.default_download_directory(::NormalizedBathymetry) = pwd()

#####
##### Shared geometry: read the regular lat–lon mesh back from the normalized file
#####

function interfaces_from_centers(centers)
    spacing1 = centers[2] - centers[1]
    spacinge = centers[end] - centers[end-1]
    interfaces = zeros(length(centers) + 1)
    interfaces[1] = centers[1] - spacing1 / 2
    interfaces[end] = centers[end] + spacinge / 2
    interfaces[2:end-1] .= (centers[1:end-1] .+ centers[2:end]) ./ 2
    return interfaces
end

function DataWrangling.longitude_interfaces(metadata::NormalizedMetadata)
    path = DataWrangling.metadata_path(metadata)
    Dataset(path) do dataset
        return interfaces_from_centers(dataset["longitude"][:])
    end
end

function DataWrangling.latitude_interfaces(metadata::NormalizedMetadata) 
    path = DataWrangling.metadata_path(metadata)
    Dataset(path) do dataset
        return interfaces_from_centers(dataset["latitude"][:])
    end
end

function Base.size(metadata::NormalizedMetadata)
    longitude = DataWrangling.longitude_interfaces(metadata)
    latitude = DataWrangling.latitude_interfaces(metadata)
    return (length(longitude) - 1, length(latitude) - 1, 1)
end

#####
##### Shared normalization: raw field → signed bottom height written as (longitude, latitude) `z`
#####

# Reorder a 2D variable to (longitude, latitude) regardless of the on-disk dimension order.
function as_longitude_latitude(variable)
    dims = NCDatasets.dimnames(variable)
    data = variable[:, :]
    return first(dims) == "longitude" ? data : permutedims(data, (2, 1))
end

function write_normalized_bathymetry(output_path, longitude, latitude, bottom_height)
    Dataset(output_path, "c") do out
        defDim(out, "longitude", length(longitude))
        defDim(out, "latitude",  length(latitude))
        defVar(out, "longitude", longitude, ("longitude",))
        defVar(out, "latitude",  latitude,  ("latitude",))
        defVar(out, "z", bottom_height, ("longitude", "latitude"))
    end
    return output_path
end

# raw → signed bottom height (negative below sea level, positive over land). `missing` → land.
to_bottom_height(::MEDSEABathymetry,  depth)     = ismissing(depth)     ? land_height : -Float64(depth)
to_bottom_height(::EMODnetBathymetry, elevation) = ismissing(elevation) ? land_height :  Float64(elevation)

function normalize_bathymetry(dataset, raw_variable, raw_path, output_path)
    Dataset(raw_path) do raw
        longitude = Float64.(raw["longitude"][:])
        latitude  = Float64.(raw["latitude"][:])
        field     = as_longitude_latitude(raw[raw_variable])
        bottom_height = map(value -> to_bottom_height(dataset, value), field)
        write_normalized_bathymetry(output_path, longitude, latitude, bottom_height)
    end
    return output_path
end

#####
##### MEDSEA — Copernicus Marine `deptho`
#####

const medsea_dataset_id   = "cmems_mod_med_phy_anfc_4.2km_static"
const medsea_dataset_part = "bathy"
const medsea_variable     = "deptho"

DataWrangling.metadata_filename(::MEDSEABathymetry, name, date, region) = "medsea_bottom_height.nc"
medsea_raw_filename() = "medsea_deptho_raw.nc"

function copernicus_credentials()
    username = get(ENV, "COPERNICUS_USERNAME", nothing)
    password = get(ENV, "COPERNICUS_PASSWORD", nothing)
    if isnothing(username) || isnothing(password)
        @warn "No Copernicus credentials found. Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD to \
               download the MEDSEA bathymetry. Free registration: https://data.marine.copernicus.eu/register."
    end
    return username, password
end

"""
    Downloads.download(metadatum::Metadatum{<:MEDSEABathymetry})

Download the Copernicus MEDSEA static bathymetry and write the normalized signed bottom-height file at
`metadata_path(metadatum)`. Idempotent.
"""
function Downloads.download(metadatum::MEDSEAMetadatum)
    output_path = metadata_path(metadatum)
    isfile(output_path) && return output_path

    username, password = copernicus_credentials()
    raw_path = joinpath(metadatum.dir, medsea_raw_filename())

    @root begin
        if !isfile(raw_path)
            @info "Downloading MEDSEA bathymetry ($(medsea_dataset_id)/$(medsea_dataset_part)) to $(metadatum.dir)..."
            credential_kw = isnothing(username) || isnothing(password) ? NamedTuple() : (; username, password)
            CopernicusMarine.subset(; dataset_id = medsea_dataset_id,
                                      dataset_part = medsea_dataset_part,
                                      variables = CopernicusMarine.pylist([medsea_variable]),
                                      output_directory = metadatum.dir,
                                      output_filename = medsea_raw_filename(),
                                      coordinates_selection_method = "outside",
                                      skip_existing = true,
                                      credential_kw...)
        end
        normalize_bathymetry(metadatum.dataset, medsea_variable, raw_path, output_path)
    end

    return output_path
end

#####
##### EMODnet — open-access ERDDAP griddap, subset to the metadata's bounding box
#####

const emodnet_griddap = "https://erddap.emodnet.eu/erddap/griddap/bathymetry_dtm_2024.nc"
const emodnet_variable = "elevation"

DataWrangling.metadata_filename(::EMODnetBathymetry, name, date, region) = "emodnet_bottom_height.nc"
emodnet_raw_filename() = "emodnet_elevation_raw.nc"

function emodnet_url(region)
    isnothing(region) && throw(ArgumentError(
        "EMODnetBathymetry requires a `region` BoundingBox — the full European DTM is too large to download whole."))
    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    return string(emodnet_griddap, "?", emodnet_variable,
                  "[(", φ₁, "):(", φ₂, ")][(", λ₁, "):(", λ₂, ")]")
end

"""
    Downloads.download(metadatum::Metadatum{<:EMODnetBathymetry})

Download the EMODnet DTM 2024 bathymetry over `metadatum.region` from the EMODnet ERDDAP server and
write the normalized signed bottom-height file at `metadata_path(metadatum)`. Idempotent. No credentials.
"""
function Downloads.download(metadatum::EMODnetMetadatum)
    output_path = metadata_path(metadatum)
    isfile(output_path) && return output_path

    raw_path = joinpath(metadatum.dir, emodnet_raw_filename())

    @root begin
        if !isfile(raw_path)
            @info "Downloading EMODnet DTM 2024 bathymetry over $(metadatum.region) to $(metadatum.dir)..."
            Downloads.download(emodnet_url(metadatum.region), raw_path)
        end
        normalize_bathymetry(metadatum.dataset, emodnet_variable, raw_path, output_path)
    end

    return output_path
end

end # module BathymetryDatasets

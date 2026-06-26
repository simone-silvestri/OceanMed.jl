module RiverForcing

export MediterraneanPrescribedLand

using Dates
using Downloads
using NCDatasets
using Oceananigans
using NumericalEarth.Lands
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Architectures: AbstractArchitecture
using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum
using OceanMed.BathymetryDatasets: MEDSEABathymetry, interfaces_from_centers

abstract type AbstractRiverRunoff end

const RiverMetadata  = Metadata{<:AbstractRiverRunoff}
const RiverMetadatum = Metadatum{<:AbstractRiverRunoff}

dardanelles_inflow1 = [0.115501732, 0.15197596, 0.202634618, 0.253293276, 0.289767504, 0.303951919, 0.289767504, 0.253293276, 0.202634618, 0.15197596, 0.115501732, 0.101317309]
dardanelles_inflow2 = [0.115572236, 0.152068734, 0.202758297, 0.25344789, 0.289944381, 0.304137468, 0.289944381, 0.25344789, 0.202758297, 0.152068734, 0.115572236, 0.101379149]
dardanelles_inflow3 = [0.115642883, 0.152161688, 0.202882245, 0.253602803, 0.290121615, 0.304323375, 0.290121615, 0.253602803, 0.202882245, 0.152161688, 0.115642883, 0.101441123]

const month_anchors = dayofyear.(DateTime.(2004, 1:12, 15))

function interpolate_monthly(monthly, date)
    d = dayofyear(date)
    if d < first(month_anchors)
        d₀, d₁, v₀, v₁ = last(month_anchors) - 366, first(month_anchors), monthly[12], monthly[1]
    elseif d ≥ last(month_anchors)
        d₀, d₁, v₀, v₁ = last(month_anchors), first(month_anchors) + 366, monthly[12], monthly[1]
    else
        m = searchsortedlast(month_anchors, d)
        d₀, d₁, v₀, v₁ = month_anchors[m], month_anchors[m+1], monthly[m], monthly[m+1]
    end
    return v₀ + (v₁ - v₀) * (d - d₀) / (d₁ - d₀)
end

#####
##### River Runoff
#####

struct RiverRunoff <: AbstractRiverRunoff end

const RiverRunoffMetadata  = Metadata{<:RiverRunoff}
const RiverRunoffMetadatum = Metadatum{<:RiverRunoff}

DataWrangling.all_dates(::RiverRunoff, var) = DateTime(2004, 1, 1) : Day(1) : DateTime(2004, 12, 31)
DataWrangling.is_three_dimensional(::RiverRunoffMetadata) = false
DataWrangling.default_inpainting(::RiverRunoffMetadata) = nothing

DataWrangling.metadata_url(m::RiverRunoffMetadata) = "https://www.dropbox.com/scl/fi/o8c3cvrl6y5n6fa6wye8k/runoff_1d_nomask_ly_climatology_efasv5.nc?rlkey=xi1irp6bz1vo293kd6iygwvet&st=5td4z9b1&dl=0"

function DataWrangling.longitude_interfaces(metadata::RiverRunoffMetadata) 
    path = DataWrangling.metadata_path(metadata)
    Dataset(path) do dataset
        return interfaces_from_centers(dataset["nav_lon"][:, 1])
    end
end

function DataWrangling.latitude_interfaces(metadata::RiverRunoffMetadata) 
    path = DataWrangling.metadata_path(metadata)
    Dataset(path) do dataset
        return interfaces_from_centers(dataset["nav_lat"][1, :])
    end
end

DataWrangling.default_download_directory(::RiverRunoff) = "./"

DataWrangling.metadata_filename(::RiverRunoff, name, date::DateTime, region) = "runoff_1d_nomask_ly_climatology_efasv5.nc"
DataWrangling.longitude_name(::RiverRunoffMetadata) = "nav_lon"
DataWrangling.latitude_name(::RiverRunoffMetadata) = "nav_lat"

function Downloads.download(meta::RiverRunoffMetadata)
    output_path = DataWrangling.metadata_path(first(meta))
    isfile(output_path) && return output_path
    mkpath(dirname(output_path))
    url = DataWrangling.metadata_url(meta)
    Downloads.download(url, output_path; progress=DataWrangling.DownloadProgress())
    return output_path
end

function Base.size(metadata::RiverRunoffMetadata)
    longitude = DataWrangling.longitude_interfaces(metadata)
    latitude = DataWrangling.latitude_interfaces(metadata)
    return (length(longitude) - 1, length(latitude) - 1, 1)
end

RiverRunoff_dataset_variable_names = Dict(
    :freshwater_flux => "sorunoff"
)

DataWrangling.dataset_variable_name(data::RiverRunoffMetadata) = RiverRunoff_dataset_variable_names[data.name]

function DataWrangling.retrieve_data(metadata::RiverRunoffMetadatum)
    path = DataWrangling.metadata_path(metadata)
    name = DataWrangling.dataset_variable_name(metadata)

    day = dayofyear(metadata.dates)
    ds = Dataset(path)
    data = ds[name][:, :, day]
    close(ds)

    data[1064, 236] = interpolate_monthly(dardanelles_inflow1, metadata.dates)
    data[1064, 237] = interpolate_monthly(dardanelles_inflow2, metadata.dates)
    data[1064, 238] = interpolate_monthly(dardanelles_inflow3, metadata.dates)

    return data
end

const dardanelles_outlets = ((1064, 236), (1064, 237), (1064, 238))

function runoff_outlet_indices(discharge)
    grid = discharge.grid

    runoff  = Array(interior(discharge))[:, :, 1]
    indices = findall(q -> !iszero(q) & !isnan(q), runoff)
    cells   = Set((I[1], I[2]) for I in indices)
    union!(cells, dardanelles_outlets)

    outlet_i = [c[1] for c in cells]
    outlet_j = [c[2] for c in cells]

    λc = Array(λnodes(grid, Center(), Center(), Center()))
    φc = Array(φnodes(grid, Center(), Center(), Center()))
    outlet_λ = [λc[i] for i in outlet_i]
    outlet_φ = [φc[j] for j in outlet_j]

    return outlet_i, outlet_j, outlet_λ, outlet_φ
end

function MediterraneanPrescribedLand(arch::AbstractArchitecture; dir = "./", kwargs...)
    grid = DataWrangling.native_grid(Metadata(:freshwater_flux; dataset = RiverRunoff(), dir))
    return MediterraneanPrescribedLand(grid)
end

function MediterraneanPrescribedLand(grid::AbstractGrid;
                                     dir = "./",
                                     time_indices_in_memory = 12,
                                     time_indexing = Oceananigans.OutputReaders.Cyclical(),
                                     freshwater_density = 1000,
                                     maximum_search_radius = 5,
                                     other_kw...)

    arch = Oceananigans.Grids.architecture(grid)
    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    river_runoff = FieldTimeSeries(Metadata(:freshwater_flux; dataset = RiverRunoff(), dir), arch; kw...)

    snapshot = river_runoff[1]
    outlet_i, outlet_j, outlet_λ, outlet_φ = runoff_outlet_indices(snapshot)
    routing = Lands.build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ; freshwater_density, maximum_search_radius)

    freshwater_flux = (; river_runoff)

    return Lands.PrescribedLand(freshwater_flux; river_routing = routing)
end

end
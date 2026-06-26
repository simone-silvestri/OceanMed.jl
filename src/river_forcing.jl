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
using OceanMed.BathymetryDatasets: MEDSEABathymetry

abstract type AbstractRiverRunoff end

const RiverMetadata  = Metadata{<:AbstractRiverRunoff}
const RiverMetadatum = Metadatum{<:AbstractRiverRunoff}

dardanelles_inflow1 = [0.115501732, 0.15197596, 0.202634618, 0.253293276, 0.289767504, 0.303951919, 0.289767504, 0.253293276, 0.202634618, 0.15197596, 0.115501732, 0.101317309]
dardanelles_inflow2 = [0.115572236, 0.152068734, 0.202758297, 0.25344789, 0.289944381, 0.304137468, 0.289944381, 0.25344789, 0.202758297, 0.152068734, 0.115572236, 0.101379149]
dardanelles_inflow3 = [0.115642883, 0.152161688, 0.202882245, 0.253602803, 0.290121615, 0.304323375, 0.290121615, 0.253602803, 0.202882245, 0.152161688, 0.115642883, 0.101441123]

#####
##### River Runoff
#####

struct RiverRunoff <: AbstractRiverRunoff end

const RiverRunoffMetadata  = Metadata{<:RiverRunoff}
const RiverRunoffMetadatum = Metadatum{<:RiverRunoff}

DataWrangling.all_dates(::RiverRunoff, var) = DateTime(1900, 1, 1) : Month(1) : DateTime(1900, 12, 1)
DataWrangling.is_three_dimensional(::RiverRunoffMetadata) = false
DataWrangling.default_inpainting(::RiverRunoffMetadata) = nothing

function interfaces_from_centers(centers)
    spacing1 = centers[2] - centers[1]
    spacinge = centers[end] - centers[end-1]
    interfaces = zeros(length(centers) + 1)
    interfaces[1] = centers[1] - spacing1 / 2
    interfaces[end] = centers[end] + spacinge / 2
    interfaces[2:end-1] .= (centers[1:end-1] .+ centers[2:end]) ./ 2
    return interfaces
end

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

    # NetCDF shenanigans
    ds = Dataset(path)
    data = ds[name][:, :, 1]
    close(ds)
    
    month = Dates.month(metadata.dates)
    data[1064, 236, 1] = dardanelles_inflow1[month]
    data[1064, 237, 1] = dardanelles_inflow2[month]
    data[1064, 238, 1] = dardanelles_inflow3[month]

    return data
end

function MediterraneanPrescribedLand(arch::AbstractArchitecture; dir = "./", kwargs...)
    grid = DataWrangling.native_grid(Metadata(:dardanelles_inflow; dataset = RiverRunoff(), dir))
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
    outlet_i, outlet_j, outlet_λ, outlet_φ = Lands.coastal_outlet_indices(snapshot)
    routing = Lands.build_river_routing(grid, outlet_i, outlet_j, outlet_λ, outlet_φ; freshwater_density, maximum_search_radius)

    freshwater_flux = (; river_runoff)

    return Lands.PrescribedLand(freshwater_flux; river_routing = routing)
end

end
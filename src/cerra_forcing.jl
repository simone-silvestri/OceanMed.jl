module CERRAForcing

export CERRAReanalysis, cerra_native_grid, CERRAPrescribedAtmosphere, CERRAPrescribedRadiation

using Dates
using Printf
using NCDatasets
using CDSAPI
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: Cyclical

using NumericalEarth
using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum, metadata_path
using NumericalEarth.Radiations: SurfaceRadiationProperties, default_stefan_boltzmann_constant

import Downloads
import NumericalEarth.DataWrangling: metadata_filename, dataset_variable_name, all_dates,
                                     first_date, last_date, default_download_directory,
                                     default_inpainting

#####
##### CERRA dataset — Copernicus European Regional ReAnalysis, single levels (~5.5 km)
#####
##### CERRA lives on a native Lambert Conformal Conic grid (1069×1069, centred on (8°E, 50°N), standard
##### parallel 50°N, 5.5 km spacing). Surface fields come in two flavours on the Climate Data Store:
##### winds/temperature/humidity/pressure as `analysis`, radiation and precipitation as `forecast`.
##### Downloading requires CDS credentials in `~/.cdsapirc` and a one-time acceptance of the CERRA
##### licence on the CDS website.
#####
##### The prescribed-atmosphere `FieldTimeSeries` are kept on this native Lambert grid; the conservative
##### regridding onto the exchange grid happens at coupling time (see `regrid_cerra_state.jl`).
#####

struct CERRAReanalysis end

const CERRAMetadata{D} = Metadata{<:CERRAReanalysis, D}
const CERRAMetadatum   = Metadatum{<:CERRAReanalysis}

const cerra_dataset_id = "reanalysis-cerra-single-levels"

const CERRA_N       = 1069
const CERRA_spacing = 5500.0
const CERRA_radius  = 6371229.0   # GRIB spherical earth

# internal variable => (CDS variable string, NetCDF short name, product_type, leadtime_hour)
# Analysis fields are instantaneous; the `forecast` fields (radiation, precipitation) are accumulated
# over the lead time, so they are de-accumulated to a rate on read (see `conversion_factor`).
const cerra_variables = Dict(
    :temperature                     => ("2m_temperature",                      "t2m",    "analysis", nothing),
    :surface_pressure                => ("surface_pressure",                    "sp",     "analysis", nothing),
    :wind_speed                      => ("10m_wind_speed",                      "si10",   "analysis", nothing),
    :wind_direction                  => ("10m_wind_direction",                  "wdir10", "analysis", nothing),
    :relative_humidity               => ("2m_relative_humidity",                "r2",     "analysis", nothing),
    :total_precipitation             => ("total_precipitation",                 "tp",     "forecast", 1),
    :snowfall                        => ("snow_fall_water_equivalent",          "sf",     "forecast", 1),
    :downwelling_shortwave_radiation => ("surface_solar_radiation_downwards",   "ssrd",   "forecast", 1),
    :downwelling_longwave_radiation  => ("surface_thermal_radiation_downwards", "strd",   "forecast", 1),
)

cds_variable(name)   = cerra_variables[name][1]
nc_short_name(name)  = cerra_variables[name][2]
cerra_product(name)  = cerra_variables[name][3]
cerra_leadtime(name) = cerra_variables[name][4]

# De-accumulate forecast products: accumulated J m⁻² → W m⁻² and accumulated kg m⁻² → kg m⁻² s⁻¹,
# dividing by the lead-time interval in seconds. Analysis fields are returned unchanged.
conversion_factor(name) = cerra_product(name) == "forecast" ? 1 / (cerra_leadtime(name) * 3600) : 1

dataset_variable_name(metadata::CERRAMetadata) = nc_short_name(metadata.name)

# CERRA analysis fields cover the whole domain (no missing values), so no inpainting is needed.
default_inpainting(::CERRAMetadata) = nothing

# Surface (2D) fields on the curvilinear grid, no vertical axis.
DataWrangling.is_three_dimensional(::CERRAMetadata) = false
DataWrangling.reversed_vertical_axis(::CERRAReanalysis) = false
Oceananigans.Fields.location(::CERRAMetadata) = (Center, Center, Center)

# CERRA analysis is 3-hourly (00, 03, …, 21 UTC).
all_dates(::CERRAReanalysis, variable) = DateTime(1984, 9, 1):Hour(3):DateTime(2021, 1, 1)
first_date(::CERRAReanalysis, variable) = first(all_dates(CERRAReanalysis(), variable))
last_date(::CERRAReanalysis, variable)  = last(all_dates(CERRAReanalysis(), variable))

Base.size(::CERRAReanalysis, variable) = (CERRA_N, CERRA_N, 1)

default_download_directory(::CERRAReanalysis) = pwd()
metadata_filename(::CERRAReanalysis, name, date::DateTime, region) = string("cerra_", nc_short_name(name), "_", Dates.format(date, "yyyymmddHH"), ".nc")

#####
##### Download from the Copernicus Climate Data Store
#####

"""
    Downloads.download(metadatum::Metadatum{<:CERRAReanalysis})

Retrieve a single CERRA field at `metadatum.dates` from the CDS as NetCDF, returning the file path.
Idempotent. Requires `~/.cdsapirc` and a one-time CERRA licence acceptance on the CDS website.
"""
function Downloads.download(metadatum::CERRAMetadatum)
    output = metadata_path(metadatum)
    isfile(output) && return output
    mkpath(metadatum.dir)

    name = metadatum.name
    date = metadatum.dates
    request = Dict{String, Any}(
        "variable"     => cds_variable(name),
        "level_type"   => "surface_or_atmosphere",
        "data_type"    => "reanalysis",
        "product_type" => cerra_product(name),
        "year"         => @sprintf("%04d", Dates.year(date)),
        "month"        => @sprintf("%02d", Dates.month(date)),
        "day"          => @sprintf("%02d", Dates.day(date)),
        "time"         => @sprintf("%02d:00", Dates.hour(date)),
        "data_format"  => "netcdf",
    )
    lt = cerra_leadtime(name)
    isnothing(lt) || (request["leadtime_hour"] = string(lt))

    @info "Downloading CERRA $(cds_variable(name)) at $date from the CDS..."
    retrieve_with_retries(cerra_dataset_id, request, output)
    return output
end

# The CDS frequently resets long-lived connections while a request sits in the queue, and
# `CDSAPI.retrieve` polls the job status with an unguarded `HTTP.request`. Re-issuing the same request
# is safe: the CDS caches identical requests, so a retry resumes the queued job rather than recomputing.
function retrieve_with_retries(dataset, request, output; maximum_attempts=10, retry_delay=30)
    for attempt in 1:maximum_attempts
        try
            return CDSAPI.retrieve(dataset, request, output)
        catch error
            error isa CDSAPI.HTTP.RequestError || rethrow(error)
            attempt == maximum_attempts && rethrow(error)
            @warn "CDS connection dropped (attempt $attempt/$maximum_attempts), retrying in $retry_delay s" exception=error
            sleep(retry_delay)
        end
    end
end

# Multi-date metadata: download each date individually (the single-date Metadatum method above is more
# specific, so iterating a Metadata dispatches to it).
function Downloads.download(metadata::CERRAMetadata)
    for metadatum in metadata
        Downloads.download(metadatum)
    end
    return metadata_path(metadata)
end

#####
##### Native Lambert grid
#####

"""
    DataWrangling.native_grid(::CERRAMetadata, arch=CPU(); halo=(3, 3, 3))

Return CERRA's native Lambert Conformal Conic grid as an Oceananigans `LambertConformalConicGrid`
(an `OrthogonalSphericalShellGrid`). Its longitude/latitude reproduce the CDS grid exactly.
"""
function DataWrangling.native_grid(::CERRAMetadata, arch = CPU(); halo = (3, 3, 3))
    ext = CERRA_N * CERRA_spacing / 2
    return LambertConformalConicGrid(arch; size = (CERRA_N, CERRA_N, 1),
                                     x = (-ext, ext), y = (-ext, ext),
                                     standard_parallel = 50, central_longitude = 8,
                                     latitude_of_origin = 50, radius = CERRA_radius,
                                     z = (0, 1), halo = halo)
end

# Read a CERRA field directly onto its native curvilinear grid. NumericalEarth's generic
# `set!(::Field, ::Metadatum)` interpolates the data from the native grid to the target with fractional
# indices, which are undefined for a curvilinear `OrthogonalSphericalShellGrid`. Because CERRA's
# `FieldTimeSeries` already live on the native grid (target == native), we override `set!` to copy the
# raw data straight in — the same mechanism JRA55 uses to customize its read. This lets the standard
# `FieldTimeSeries(metadata, arch)` constructor build CERRA series on the native Lambert grid.
# The CDS NetCDF short name is occasionally not the one we expect; fall back to the single 2D data
# variable in the file (CERRA files carry one field plus latitude/longitude/valid_time/expver).
function cerra_nc_variable(dataset, metadatum)
    name = dataset_variable_name(metadatum)
    haskey(dataset, name) && return name
    coordinates = ("latitude", "longitude", "valid_time", "time", "x", "y", "expver")
    for key in keys(dataset)
        key in coordinates && continue
        ndims(dataset[key]) ≥ 2 && return key
    end
    return name
end

function Oceananigans.Fields.set!(target::Field, metadatum::CERRAMetadatum; kw...)
    Downloads.download(metadatum)
    factor = Float32(conversion_factor(metadatum.name))
    data = Dataset(metadata_path(metadatum)) do ds
        factor .* Float32.(coalesce.(ds[cerra_nc_variable(ds, metadatum)][:, :, 1], NaN))
    end
    interior(target)[:, :, 1] .= on_architecture(architecture(target.grid), data)
    fill_halo_regions!(target)
    return target
end

"""
    CERRAPrescribedAtmosphere(architecture = CPU();
                              start_date = first_date(CERRAReanalysis(), :temperature),
                              end_date = last_date(CERRAReanalysis(), :temperature),
                              dir = ".",
                              time_indices_in_memory = 10,
                              time_indexing = Cyclical(),
                              surface_layer_height = 10,
                              region = nothing,
                              other_kw...)

Build a `PrescribedAtmosphere` forced by CERRA, whose `FieldTimeSeries` live on CERRA's **native
Lambert grid**. Reads CERRA's analysis fields (10 m wind speed/direction, 2 m temperature, 2 m relative
humidity, surface pressure) and the forecast precipitation, deriving the eastward/northward velocities,
specific humidity, and the rain/snow freshwater flux (`snow = snowfall`, `rain = total − snowfall`). The
conservative regridding onto the model exchange grid is applied at coupling time by the atmosphere
exchanger (see `regrid_cerra_state.jl`).

Mirrors [`JRA55PrescribedAtmosphere`](@ref). `time_indices_in_memory`/`time_indexing` and any
`other_kw` are forwarded to each variable's `FieldTimeSeries`. `region` is accepted for interface
symmetry but is currently ignored — CERRA offers no spatial subsetting on retrieval, and the native
grid is always the full Lambert domain.

For the downwelling radiation, build a [`CERRAPrescribedRadiation`](@ref) and pass it as the
`radiation` to `OceanSeaIceModel`.
"""
function CERRAPrescribedAtmosphere(arch = CPU();
                                   start_date = first_date(CERRAReanalysis(), :temperature),
                                   end_date = last_date(CERRAReanalysis(), :temperature),
                                   dir = ".",
                                   time_indices_in_memory = 10,
                                   time_indexing = Cyclical(),
                                   surface_layer_height = 10,
                                   region = nothing,
                                   other_kw...)
    
    dataset = CERRAReanalysis()
    kw = merge((; time_indices_in_memory, time_indexing), other_kw)
    meta(name) = Metadata(name; dataset, dir, start_date, end_date, region)
    fts(name)  = FieldTimeSeries(meta(name), arch; kw...)   # built on the native Lambert grid (see set! above)

    # Store the RAW CERRA fields (the backed FieldTimeSeries that `update_state!` refreshes each step).
    # The eastward/northward velocities, specific humidity and rain/snow split are NONLINEAR functions of
    # these, so they are derived every timestep on the native grid — before the conservative regrid —
    # in `interpolate_state!` (see `regrid_cerra_state.jl`). The raw fields ride through the standard
    # `PrescribedAtmosphere` container by convention:
    #   velocities.u = 10 m wind speed,   velocities.v = 10 m wind direction,
    #   tracers.q    = 2 m relative humidity,
    #   freshwater_flux.rain = total precipitation,   freshwater_flux.snow = snowfall,
    # while tracers.T (temperature) and pressure are used directly. The actual liquid rain is derived as
    # total precipitation − snowfall in `interpolate_state!`.
    Ta    = fts(:temperature)
    grid  = Ta.grid
    times = Ta.times

    freshwater_flux = NumericalEarth.Atmospheres.PrescribedPrecipitationFlux(rain = fts(:total_precipitation), snow = fts(:snowfall))

    return NumericalEarth.PrescribedAtmosphere(grid, times;
                                               velocities = (u = fts(:wind_speed), v = fts(:wind_direction)),
                                               tracers = (T = Ta, q = fts(:relative_humidity)),
                                               pressure = fts(:surface_pressure),
                                               freshwater_flux,
                                               surface_layer_height)
end

"""
    CERRAPrescribedRadiation(architecture = CPU();
                             dataset = CERRAReanalysis(),
                             start_date = first_date(dataset, :downwelling_shortwave_radiation),
                             end_date = last_date(dataset, :downwelling_shortwave_radiation),
                             dir = ".",
                             time_indices_in_memory = 10,
                             time_indexing = Cyclical(),
                             ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                             sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                             snow_surface = nothing,
                             land_surface = nothing,
                             stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                             region = nothing,
                             other_kw...)

Build a `PrescribedRadiation` from CERRA's downwelling shortwave and longwave radiation (forecast
products, de-accumulated to W m⁻²), whose `FieldTimeSeries` live on CERRA's native Lambert grid. Pass
the result as `radiation` to `OceanSeaIceModel`; the conservative regridding onto the exchange grid is
applied at coupling time by the radiation exchanger (see `regrid_cerra_state.jl`).

Mirrors [`JRA55PrescribedRadiation`](@ref): the surface radiative properties (albedo, emissivity)
default to standard ocean/sea-ice values (pass `*_surface = nothing` to omit a surface).
`time_indices_in_memory`/`time_indexing` and any `other_kw` are forwarded to the radiation
`FieldTimeSeries`; `region` is accepted for symmetry but currently ignored (see
[`CERRAPrescribedAtmosphere`](@ref)).
"""
function CERRAPrescribedRadiation(arch = CPU();
                                  dataset = CERRAReanalysis(),
                                  start_date = first_date(dataset, :downwelling_shortwave_radiation),
                                  end_date = last_date(dataset, :downwelling_shortwave_radiation),
                                  dir = ".",
                                  time_indices_in_memory = 10,
                                  time_indexing = Cyclical(),
                                  ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                                  sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                                  snow_surface = nothing,
                                  land_surface = nothing,
                                  stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                                  region = nothing,
                                  other_kw...)

    kw = merge((; time_indices_in_memory, time_indexing), other_kw)
    meta(name) = Metadata(name; dataset, dir, start_date, end_date, region)
    shortwave = FieldTimeSeries(meta(:downwelling_shortwave_radiation), arch; kw...)
    longwave  = FieldTimeSeries(meta(:downwelling_longwave_radiation),  arch; kw...)

    return NumericalEarth.Radiations.PrescribedRadiation(shortwave, longwave;
                                                         ocean_surface, sea_ice_surface,
                                                         snow_surface, land_surface,
                                                         stefan_boltzmann_constant)
end

include("regrid_cerra_state.jl")

end # module CERRAForcing

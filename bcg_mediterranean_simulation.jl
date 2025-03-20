# # Mediterranean simulation with restoring to ECCO
#
# This example is a comprehensive example of setting up and running a high-resolution ocean
# simulation for the Mediterranean Sea using the Oceananigans and ClimaOcean packages, with
# a focus on restoring temperature and salinity fields from the ECCO (Estimating the Circulation
# and Climate of the Ocean) dataset. 
#
# The example is divided into several sections, each handling a specific part of the simulation
# setup and execution process.

# ## Initial Setup with Package Imports
#
# We begin by importing necessary Julia packages for visualization (CairoMakie), ocean modeling
# (Oceananigans, ClimaOcean), and handling of dates and times (CFTime, Dates). 
# These packages provide the foundational tools for creating the simulation environment, 
# including grid setup, physical processes modeling, and data visualization.

# NOT WORKING ON XQUARTZ using GLMakie
using Pkg
using CairoMakie
using Oceananigans
using Oceananigans.Grids
using Oceananigans: architecture
using ClimaOcean
using ClimaOcean.ECCO
using ClimaOcean.ECCO: ECCO4Monthly
using Oceananigans.Units
using Printf
using NCDatasets
using Dates

# ##
using OceanBiome

include("vertical_diffusivity.jl")

# ## Grid Configuration for the Mediterranean Sea
#
# The script defines a high-resolution grid to represent the Mediterranean Sea, specifying the domain in terms of longitude (λ₁, λ₂), 
# latitude (φ₁, φ₂), and a stretched vertical grid to capture the depth variation (`z_faces`). 
# The grid resolution is set to approximately 1/15th of a degree, which translates to a spatial resolution of about 7 km. 
# This section demonstrates the use of the LatitudeLongitudeGrid function to create a grid that matches the
# Mediterranean's geographical and bathymetric features.

arch = CPU()

const λ₁, λ₂  = (-8.6, 42) # domain in longitude
const φ₁, φ₂  = (  30, 48) # domain in latitude

Nx = 30 * ceil(Int, λ₂ - λ₁) # 1/50th of a degree resolution
Ny = 30 * ceil(Int, φ₂ - φ₁) # 1/50th of a degree resolution
Nz = 140 # 140 vertical levels

# Probably you want to change `r_faces` to get the resolution you want 
# at surface vs depth. This is an Array of size `Nz+1` that defines the 
# position of the initial position of z-interfaces (when `η = 0`)
ds = Dataset("data/zc_copernicus.nc")
r_centers = reverse(ds["depth"][:])
r_faces   = zeros(Nz+1)
for k in Nz:-1:1
  r_faces[k] = r_faces[k+1] - (r_centers[k] + r_faces[k+1]) * 2
end

z_faces = MutableVerticalDiscretization(r_faces)

# To run on Distributed architectures (for example 4 ranks in x and 4 in y):
# arch = Distributed(arch, partition = Partition(x = 4, y = 4))

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude  = (φ₁, φ₂),
                             longitude = (λ₁, λ₂),
                             z = z_faces,
                             halo = (7, 7, 7))

# ### Bathymetry Interpolation
#
# The script interpolates bathymetric data onto the grid, ensuring that the model accurately represents 
# the sea floor's topography. Parameters such as `minimum_depth` and `interpolation_passes`
# are adjusted to refine the bathymetry representation.

bottom_height = regrid_bathymetry(grid, 
                                  minimum_depth = 5,
                                  interpolation_passes = 1,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# ## ECCO Restoring
#
# The model is restored in a sponge region outside Gibraltar to the temperature and salinity fields from the ECCO dataset.
# We build the restoring using the `ECCORestoring` functionality. 
# This allows us to nudge the model towards realistic temperature and salinity profiles.
# `ECCORestoring` accepts a `mask` keyword argument to restrict the restoring region.

const λₑ = - 7 # eastern bound of the restoring region

@inline gibraltar_mask(λ, φ, z, t) = min(1.0, max(0.0, 1 / (λ₁ - λₑ) * (λ - λₑ)))

dates = DateTime(1993, 1, 1) : Month(1) : DateTime(1993, 5, 1)

# This contructor downloads the ECCO dataset in the `dates` range. Make sure you have internet access 
# and you pass login information to the ECCO donwloader (see https://github.com/CliMA/ClimaOcean.jl/blob/main/src/DataWrangling/ECCO/README.md) 
# If you have other data to use as restoring we can add a custom backend to use the data.
FT = ECCORestoring(:temperature, arch; dates, mask=gibraltar_mask, rate=1/10days)
FS = ECCORestoring(:salinity, arch;    dates, mask=gibraltar_mask, rate=1/10days)

# Constructing the Simulation
#
# We construct an ocean simulation that evolves two tracers, temperature (:T), salinity (:S)
# and we pass the previously defined forcing that nudge these tracers 

momentum_advection = WENOVectorInvariant()
tracer_advection = WENO(order=7)
closure = nothing
timestepper = :SplitRungeKutta3

# ## Biogeochemistry test: fdattilo
using Oceananigans.Biogeochemistry: AbstractBiogeochemistry

#= struct DtlBiogeochemistry{P1, P2, P3} <: AbstractBiogeochemistry
    par1 :: P1
    par2 :: P2
    par3 :: P3
end 

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers

required_biogeochemical_tracers(::DtlBiogeochemistry) = (:ph, :alk)=#
# ## Setting the biogeochemistry
bgc = LOBSTER(; grid,
                carbonate = false, # set true if you want CO₂ boundary conditions
                oxygen = false,

)
# ## consider to ad CO₂_flux  as boundary condition
# CO₂_flux = CarbonDioxideGasExchangeBoundaryCondition()
ocean = ocean_simulation(grid; 
                         timestepper, 
                         momentum_advection, 
                         tracer_advection, 
                         closure, 
                         biogeochemistry = bgc,#DtlBiogeochemistry(1, 2, 3),
                         forcing=(T=FT, S=FS)#,
                         #boundary_conditions = (DIC = FieldBoundaryConditions(top = CO₂_flux), )
                         )

####
#### Initialize the ph and alkalinity
####

# load the dataset
ds  = Dataset("data/20250101_m-OGS--NUTR-MedBFM4-MED-b20250211_an-sv09.00.nc")
dp  = Dataset("data/20250101_m-OGS--PFTC-MedBFM4-MED-b20250211_an-sv09.00.nc")
dd  = Dataset("data/20170603_d-OGS--CARB-MedBFM3-MED-b20221101_re-sv05.00.nc")
λ   = ds["longitude"][:]
φ   = ds["latitude"][:]
z   = reverse(ds["depth"][:])

nutrients_data = reverse(ds["no3"][:, :, :],  dims=3)
phyto_data = reverse(dp["phyc"][:, :, :],  dims=3)
zoo_data = reverse(dp["zooc"][:, :, :],  dims=3)
detritus_data = reverse(dd["detritus"][:, :, :],  dims=3)
#ph_data  = reverse(ds["ph"][:, :, :],  dims=3)
#alk_data = reverse(ds["talk"][:, :, :], dims=3)

# creating a grid for interpolation
Nλ = length(λ)
Nφ = length(φ)
Nz = length(z)

z_faces   = zeros(Nz+1)
for k in Nz:-1:1
    z_faces[k] = z_faces[k+1] - (z[k] + z_faces[k+1]) * 2
end


λᴮ₁ = λ[1]  - (λ[2]  - λ[1]   ) / 2
λᴮ₂ = λ[Nλ] + (λ[Nλ] - λ[Nλ-1]) / 2

φᴮ₁ = φ[1]  - (φ[2]  - φ[1]   ) / 2
φᴮ₂ = φ[Nφ] + (φ[Nφ] - φ[Nφ-1]) / 2

bcg_grid = LatitudeLongitudeGrid(size = (Nλ, Nφ, Nz),
                                 latitude  = (φᴮ₁, φᴮ₂),
                                 longitude = (λᴮ₁, λᴮ₂),
                                 z = z_faces)

# creating bgc fields

Ni = CenterField(bcg_grid)
Pi = CenterField(bcg_grid)
Zi = CenterField(bcg_grid)
Di = CenterField(bcg_grid)
#ph  = CenterField(bcg_grid)
#alk = CenterField(bcg_grid)

# masking fields
nutrients_data[ismissing.(phyto_data)]   .= NaN
phyto_data[ismissing.(phyto_data)]   .= NaN
zoo_data[ismissing.(phyto_data)]   .= NaN
detritus_data[ismissing.(phyto_data)]   .= NaN
#ph_data[ismissing.(ph_data)]   .= NaN
#alk_data[ismissing.(alk_data)] .= NaN

nutrients_data = nutrients_data .|> Float64
phyto_data  = phyto_data .|> Float64
zoo_data = zoo_data .|> Float64
detritus_data  = detritus_data .|> Float64
#ph_data  = ph_data .|> Float64
#alk_data = alk_data .|> Float64

set!(Ni, nutrients_data)
set!(Pi,  phyto_data)
set!(Zi, zoo_data)
set!(Di,  detritus_data)
#set!(ph,  ph_data)
#set!(alk, alk_data)

mask = CenterField(bcg_grid, Bool)
mask .= isnan.(interior(Ni))
mask .= isnan.(interior(Pi))
mask .= isnan.(interior(Zi))
mask .= isnan.(interior(Di))
#mask .= isnan.(interior(ph))

# we need to inpaint the fields to avoid problems at the boundaries
ClimaOcean.DataWrangling.inpaint_mask!(Ni,  mask; inpainting=10)
ClimaOcean.DataWrangling.inpaint_mask!(Pi, mask; inpainting=10)
ClimaOcean.DataWrangling.inpaint_mask!(Zi,  mask; inpainting=10)
ClimaOcean.DataWrangling.inpaint_mask!(Di, mask; inpainting=10)
#ClimaOcean.DataWrangling.inpaint_mask!(ph,  mask; inpainting=10)
#ClimaOcean.DataWrangling.inpaint_mask!(alk, mask; inpainting=10)

using Oceananigans.Fields: interpolate!
# now we can interpolate
interpolate!(ocean.model.tracers.N,  Ni)
interpolate!(ocean.model.tracers.P, Pi)
interpolate!(ocean.model.tracers.Z,  Zi)
interpolate!(ocean.model.tracers.D, Di)
#interpolate!(ocean.model.tracers.ph,  ph)
#interpolate!(ocean.model.tracers.alk, alk)

# Try to infer the data grid from the nc file


# Initializing the model
#
# The model can be initialized with custom values or with ecco fields.
# In this case, our ECCO dataset has access to a temperature and a salinity
# field, so we initialize temperature T and salinity S from ECCO.

set!(ocean.model, T=Metadata(:temperature; dates=dates[1], dataset=ECCO4Monthly()), 
                  S=Metadata(:salinity;    dates=dates[1], dataset=ECCO4Monthly()).
                  NO₃ = Ni, P = Pi, Z = Zi, DOM = Di)

## Adding an atmospheric forcing

# we use JRA55-do dataset to force the model with surface heat fluxes and wind stress
# Only 10 time instances of the JRA55 datasets are loaded in memory at each time
# these are updated as the model progresses
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(10), include_rivers_and_icebergs=true, dir="./data")

# This uses a quite simple ocean albedo model (latitude dependent) and 
# an ocean emissivity of 0.97. It is all customizable
radiation = Radiation()

# The coupled model! (we have no sea-ice so we do not add it)
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# The coupled simulation:
Δt = 10
# Δt = 4minutes
stop_time = 30days
simulation = Simulation(coupled_model; Δt, stop_time)

function progress(sim) 
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers

    @info @sprintf("Time: %s, Iteration %d, Δt %s, max(vel): (%.2e, %.2e, %.2e), max(T, S): %.2f, %.2f\n",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   maximum(abs, T), maximum(abs, S))
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

#Versione con uscite ogni 24 ore
simulation.output_writers[:surface_fields] = JLD2OutputWriter(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                              schedule = TimeInterval(24hours),
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true,
                                                              filename = "med_surface_fields.jld2")


simulation.output_writers[:checkpointer] = Checkpointer(ocean.model, 
							schedule = TimeInterval(30days),
							overwrite_existing = true,
							prefix = "mediterranean")

## Running the Simulation
run!(simulation)

# Record a video
#
# Let's read the data and record a video of the Mediterranean Sea's surface
# (1) Zonal velocity (u)
# (2) Meridional velocity (v)
# (3) Temperature (T)
# (4) Salinity (S)

u_series = FieldTimeSeries("med_surface_fields.jld2", "u"; backend=OnDisk())
v_series = FieldTimeSeries("med_surface_fields.jld2", "v"; backend=OnDisk())
T_series = FieldTimeSeries("med_surface_fields.jld2", "T"; backend=OnDisk())
S_series = FieldTimeSeries("med_surface_fields.jld2", "S"; backend=OnDisk())
iter = Observable(1)

u = @lift(u_series[$iter])
v = @lift(v_series[$iter])
T = @lift(T_series[$iter])
S = @lift(S_series[$iter])

fig = Figure()
ax  = Axis(fig[1, 1], title = "surface zonal velocity ms⁻¹")
heatmap!(ax, u)
ax  = Axis(fig[1, 2], title = "surface meridional velocity ms⁻¹")
heatmap!(ax, v)
ax  = Axis(fig[2, 1], title = "surface temperature ᵒC")
heatmap!(ax, T)
ax  = Axis(fig[2, 2], title = "surface salinity psu")
heatmap!(ax, S)

CairoMakie.record(fig, "mediterranean_video.mp4", 1:length(u_series.times); framerate = 5) do i
    @info "recording iteration $i"
    iter[] = i    
end
# ![](mediterranean_video.mp4)

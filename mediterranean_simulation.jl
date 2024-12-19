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

using GLMakie
using Oceananigans
using Oceananigans: architecture
using ClimaOcean
using ClimaOcean.ECCO
using ClimaOcean.ECCO: ECCO4Monthly
using Oceananigans.Units
using Printf

using CFTime
using Dates

# ## Grid Configuration for the Mediterranean Sea
#
# The script defines a high-resolution grid to represent the Mediterranean Sea, specifying the domain in terms of longitude (λ₁, λ₂), 
# latitude (φ₁, φ₂), and a stretched vertical grid to capture the depth variation (`z_faces`). 
# The grid resolution is set to approximately 1/15th of a degree, which translates to a spatial resolution of about 7 km. 
# This section demonstrates the use of the LatitudeLongitudeGrid function to create a grid that matches the
# Mediterranean's geographical and bathymetric features.

λ₁, λ₂  = ( 0, 42) # domain in longitude
φ₁, φ₂  = (30, 45) # domain in latitude

Nx = 50 * Int(λ₂ - λ₁) # 1/50th of a degree resolution
Ny = 50 * Int(φ₂ - φ₁) # 1/50th of a degree resolution
Nz = length(z_faces) - 1

z_faces = exponential_z_faces(; Nz, depth=5000, h=34)

arch = CPU()

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
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# ## ECCO Restoring
#
# The model is restored in a sponge region outside Gibraltar to the temperature and salinity fields from the ECCO dataset.
# We build the restoring using the `ECCORestoring` functionality. 
# This allows us to nudge the model towards realistic temperature and salinity profiles.
# `ECCORestoring` accepts a `mask` keyword argument to restrict the restoring region.

@inline gibraltar_mask(x, y, z, t) = min(max(0, 5 - y), 1)

dates = DateTimeProlepticGregorian(1993, 1, 1) : Month(1) : DateTimeProlepticGregorian(1993, 5, 1)

# This contructor downloads the ECCO dataset in the `dates` range. Make sure you have internet access 
# and you pass login information to the ECCO donwloader (see https://github.com/CliMA/ClimaOcean.jl/blob/main/src/DataWrangling/ECCO/README.md) 
# If you have other data to use as restoring we can add a custom backend to use the data.
FT = ECCORestoring(:temperature, grid; dates, mask=gibraltar_mask, rate=1/5days)
FS = ECCORestoring(:salinity, grid;    dates, mask=gibraltar_mask, rate=1/5days)

# The velocities are restored to zero with the same rate in the mask region
@inline restore_velocity_to_zero(x, y, z, t, vel, rate) = - gibraltar_mask(x, y, z, t) * vel * rate

Fu = Forcing(restore_velocity_to_zero, field_dependencies=:u, parameters=1/5days)
Fv = Forcing(restore_velocity_to_zero, field_dependencies=:v, parameters=1/5days)

# Constructing the Simulation
#
# We construct an ocean simulation that evolves two tracers, temperature (:T), salinity (:S)
# and we pass the previously defined forcing that nudge these tracers 

ocean = ocean_simulation(grid; forcing=(T=FT, S=FS, u=Fu, v=Fv))

# Initializing the model
#
# The model can be initialized with custom values or with ecco fields.
# In this case, our ECCO dataset has access to a temperature and a salinity
# field, so we initialize temperature T and salinity S from ECCO.

set!(ocean.model, T=ECCOMetadata(:temperature; dates=dates[1]), 
                  S=ECCOMetadata(:salinity;    dates=dates[1]))


## Adding an atmospheric forcing

# we use JRA55-do dataset to force the model with surface heat fluxes and wind stress
# Only 10 time instances of the JRA55 datasets are loaded in memory at each time
# these are updated as the model progresses
atmosphere = JRA55PrescribedAtmosphere(arch; backend = JRA55NetCDFBackend(10))

# the skin temperature is computed as a balance of external and internal heat fluxes
similarity_theory = SimilarityTheoryTurbulentFluxes(grid, surface_temperature_type = SkinTemperature())

# This uses a quite simple ocean albedo model (latitude dependent) and 
# an ocean emissivity of 0.97. It is all customizable
radiation = Radiation()

# The coupled model! (we have no sea-ice so we do not add it)
coupled_model = OceanSeaIceModel(ocean; atmosphere, similarity_theory, radiation)

# The coupled simulation:
Δt = 2minutes
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

coupled_simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

## Running the Simulation
run!(simulation)

# # Record a video
# #
# # Let's read the data and record a video of the Mediterranean Sea's surface
# # (1) Zonal velocity (u)
# # (2) Meridional velocity (v)
# # (3) Temperature (T)
# # (4) Salinity (S)

# u_series = FieldTimeSeries("med_surface_field.jld2", "u"; backend=OnDisk())
# v_series = FieldTimeSeries("med_surface_field.jld2", "v"; backend=OnDisk())
# T_series = FieldTimeSeries("med_surface_field.jld2", "T"; backend=OnDisk())
# S_series = FieldTimeSeries("med_surface_field.jld2", "S"; backend=OnDisk())
# iter = Observable(1)

# u = @lift(u_series[$iter])
# v = @lift(v_series[$iter])
# T = @lift(T_series[$iter])
# S = @lift(S_series[$iter])

# fig = Figure()
# ax  = Axis(fig[1, 1], title = "surface zonal velocity ms⁻¹")
# heatmap!(ax, u)
# ax  = Axis(fig[1, 2], title = "surface meridional velocity ms⁻¹")
# heatmap!(ax, v)
# ax  = Axis(fig[2, 1], title = "surface temperature ᵒC")
# heatmap!(ax, T)
# ax  = Axis(fig[2, 2], title = "surface salinity psu")
# heatmap!(ax, S)

# CairoMakie.record(fig, "mediterranean_video.mp4", 1:length(u_series.times); framerate = 5) do i
#     @info "recording iteration $i"
#     iter[] = i    
# end
# # ![](mediterranean_video.mp4)
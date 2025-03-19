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
using MPI
MPI.Init()

using CairoMakie
using Oceananigans
using Oceananigans.Grids
using Oceananigans: architecture
using Oceananigans.Advection: FluxFormAdvection
using ClimaOcean
using ClimaOcean.ECCO
using ClimaOcean.ECCO: ECCO4Monthly
using Oceananigans.Units
using Printf
using NCDatasets
using Dates

# ## Grid Configuration for the Mediterranean Sea
#
# The script defines a high-resolution grid to represent the Mediterranean Sea, specifying the domain in terms of longitude (λ₁, λ₂), 
# latitude (φ₁, φ₂), and a stretched vertical grid to capture the depth variation (`z_faces`). 
# The grid resolution is set to approximately 1/15th of a degree, which translates to a spatial resolution of about 7 km. 
# This section demonstrates the use of the LatitudeLongitudeGrid function to create a grid that matches the
# Mediterranean's geographical and bathymetric features.

# To find the optimal partitioning strategy, cut the domain trying to have the same
# number of fluid (active) cells in the domain.
# An example of how to do this is in 
# https://github.com/simone-silvestri/OceanScalingTests.jl/blob/main/src/grid_load_balance.jl
# 
# To have unevening partitioning in Oceananigans the syntax is 
# using Oceananigans.DistributedComputations: Sizes
# s = Sizes(n1, n2, n3, n4) # For example for 4 ranks
# arch = Distributed(GPU(), partition=Partition(x = s, y = 2)) # Note!!!! n1 + n2 + n3 + n4 should be == Nx!!!
# 
# Another option is
# f = Fractional(1/3, 1/2, ...)
# arch = Distributed(GPU(), partition=Partition(x = f, y = 2))

# synchronized_communication is because vertical mixing does not work with pipelining
@info "Initializing architecture"

arch = Distributed(GPU(), partition=Partition(4, 2), synchronized_communication=true) 

const λ₁, λ₂  = (-8.6, 42) # domain in longitude
const φ₁, φ₂  = (  30, 48) # domain in latitude

Nx = 72 * ceil(Int, λ₂ - λ₁) # 1/50th of a degree resolution
Ny = 72 * ceil(Int, φ₂ - φ₁) # 1/50th of a degree resolution
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

@info "Building grid variable"

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
tracer_advection = FluxFormAdvection(high_order, high_order, low_order)
closure = CalibratedRiBasedVerticalDiffusivity()
timestepper = :SplitRungeKutta3

ocean = ocean_simulation(grid; 
                         timestepper,
                         momentum_advection,
                         tracer_advection,
                         closure,
                         forcing=(T=FT, S=FS))

# Initializing the model
#
# The model can be initialized with custom values or with ecco fields.
# In this case, our ECCO dataset has access to a temperature and a salinity
# field, so we initialize temperature T and salinity S from ECCO.

set!(ocean.model, T=Metadata(:temperature; dates=dates[1], dataset=ECCO4Monthly()), 
                  S=Metadata(:salinity;    dates=dates[1], dataset=ECCO4Monthly()))

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

wall_time = Ref(time_ns())

function progress(sim) 
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers
    rank = arch.local_rank

    step_time = (time_ns() - wall_time) * 1e-9
    @info @sprintf("Time: %s, rank: %d, step_time: %s, Iteration %d, Δt %s, max(vel): (%.2e, %.2e, %.2e), max(T, S): %.2f, %.2f\n",
                   prettytime(sim.model.clock.time),
                   rank,
		   prettytime(step_time),
		   sim.model.clock.iteration,
                   prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   maximum(abs, T), maximum(abs, S))
 
    wall_time[] = time_ns()       
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

# Versione con uscite ogni 24 ore
simulation.output_writers[:surface_fields] = JLD2OutputWriter(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                              schedule = TimeInterval(24hours),
                                                              indices = (:, :, grid.Nz),
                                                              overwrite_existing = true,
                                                              filename = "distributed_med_surface_fields.jld2")


simulation.output_writers[:checkpointer] = Checkpointer(ocean.model, 
							schedule = TimeInterval(30days),
							overwrite_existing = true,
							prefix = "distributed_mediterranean")

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

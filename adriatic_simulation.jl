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
using Oceananigans.Units
using Printf
using PythonCall
using NCDatasets
using Dates

# include("vertical_diffusivity.jl")

# ## Grid Configuration for the Mediterranean Sea
#
# The script defines a high-resolution grid to represent the Mediterranean Sea, specifying the domain in terms of longitude (λ₁, λ₂), 
# latitude (φ₁, φ₂), and a stretched vertical grid to capture the depth variation (`z_faces`). 
# The grid resolution is set to approximately 1/15th of a degree, which translates to a spatial resolution of about 7 km. 
# This section demonstrates the use of the LatitudeLongitudeGrid function to create a grid that matches the
# Mediterranean's geographical and bathymetric features.

arch = CPU()

ds = Dataset("data/mesh_mask2D_NA.nc")
xc = ds["XC"][:]
yc = ds["YC"][:]

x_faces = [xc..., xc[end] + (xc[end] - xc[end-1])]
y_faces = [yc..., yc[end] + (yc[end] - yc[end-1])]

z_faces = [-228.9898, -217.6341, -206.7042, -196.1894, -186.0793, -176.3635, 
    -167.032, -158.075, -149.4827, -141.2458, -133.3548, -125.8009, -118.575, 
    -111.6685, -105.0728, -98.77982, -92.78125, -87.06921, -81.63593, 
    -76.47385, -71.57558, -66.93387, -62.54169, -58.39215, -54.47855, 
    -50.79433, -47.33311, -44.08865, -41.05489, -38.2259, -35.59592, 
    -33.15932, -30.91062, -28.8445, -26.95574, -25.2393, -23.69024, 
    -22.30376, -21.0752, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, 
    -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0]

Nz = length(z_faces) - 1 # 140 vertical levels
Nx = length(x_faces) - 1
Ny = length(y_faces) - 1

z_faces = MutableVerticalDiscretization(z_faces)

# To run on Distributed architectures (for example 4 ranks in x and 4 in y):
# arch = Distributed(arch, partition = Partition(x = 4, y = 4))

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude  = y_faces,
                             longitude = x_faces,
                             z = z_faces,
                             halo = (7, 7, 7))

# ### Load and set the Bathymetry 

bottom_height = - ds["Depth"][:, :] 
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Load the forcing data
start_date = Date(2017, 1, 1)

bbox = ClimaOcean.DataWrangling.BoundingBox(
            longitude = (x_faces[1]-2, x_faces[end]+2), 
            latitude  = (y_faces[1]-2, y_faces[end]+2))

dir = "./data"

dataset = GLORYSDaily()
u_meta = Metadata(:u_velocity;  dataset, dir, bounding_box=bbox, start_date)
v_meta = Metadata(:v_velocity;  dataset, dir, bounding_box=bbox, start_date)
T_meta = Metadata(:temperature; dataset, dir, bounding_box=bbox, start_date)
S_meta = Metadata(:salinity;    dataset, dir, bounding_box=bbox, start_date)

# 
path = ClimaOcean.DataWrangling.metadata_path(u_meta[1])
if !isfile(path)
    throw(error("Data needs to be downloaded on the login node! run the `download_glorys_data.jl` file."))
end

# ClimaOcean.DataWrangling.download_dataset(u_meta)
# ClimaOcean.DataWrangling.download_dataset(v_meta)
# ClimaOcean.DataWrangling.download_dataset(T_meta)
# ClimaOcean.DataWrangling.download_dataset(S_meta)

#####
##### Part we need to work on:
#####

# Maybe we need to restore to this?
u_out = FieldTimeSeries(u_meta, grid; time_indices_in_memory=10)
v_out = FieldTimeSeries(v_meta, grid; time_indices_in_memory=10)

# For sure we need to restore to these!
T_out = FieldTimeSeries(T_meta, grid; time_indices_in_memory=10)
S_out = FieldTimeSeries(S_meta, grid; time_indices_in_memory=10)

# Set the boundary conditions
# TODO: Set up the correct boundary conditions
# Remember, we _need_ FluxBoundaryConditions at the `top` and drag BC at the `bottom`
# u_bcs = .... (open bc with u_out as external data)
# v_bcs = .... (open bc with v_out as external data)
#
# Note, most likely you will need to set up a sponge layer
# T_bcs = .... (restoring bc with T_out as external data)
# S_bcs = .... (restoring bc with S_out as external data)

#####
##### After this, everything should work out of the box
#####

# Constructing the Simulation
#
# We construct an ocean simulation that evolves two tracers, temperature (:T), salinity (:S)
# and we pass the previously defined forcing that nudge these tracers 

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order=7)

# Choose the correct TS that works with the bcs
# timestepper = :QuasiAdamsBashforth2
# free_surface = Choose the correct free_surface that works with the bcs

ocean = ocean_simulation(grid; 
                         momentum_advection, 
                         tracer_advection)
                         # TODO: Uncomment below...
                         # timestepper,
                         # free_surface,
                         # boundary_conditions=(u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs))

# Initializing the model
#
# The model can be initialized with custom values or with ecco fields.
# In this case, our ECCO dataset has access to a temperature and a salinity
# field, so we initialize temperature T and salinity S from ECCO.

set!(ocean.model, T=Metadatum(:temperature; date=start_date, dataset), 
                  S=Metadatum(:salinity;    date=start_date, dataset))

## Adding an atmospheric forcing

# we use JRA55-do dataset to force the model with surface heat fluxes and wind stress
# Only 100 time instances of the JRA55 datasets are loaded in memory at each time
# these are updated as the model progresses
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(100), include_rivers_and_icebergs=true, dir="./data")

# This uses a quite simple ocean albedo model (latitude dependent) and 
# an ocean emissivity of 0.97. It is all customizable
radiation = Radiation()

# The coupled model! (we have no sea-ice so we do not add it)
coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)

# The coupled simulation:
Δt = 10
stop_time = 5days
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
simulation.output_writers[:surface_fields] = JLD2Writer(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                       schedule = TimeInterval(24hours),
                                                       indices = (:, :, grid.Nz),
                                                       overwrite_existing = true,
                                                       filename = "med_surface_fields.jld2")

ocean.output_writers[:checkpointer] = Checkpointer(ocean.model, 
						   schedule = IterationInterval(86400),
						   overwrite_existing = true,
						   prefix = "mediterranean")

## Running the Simulation
run!(simulation)

simulation.Δt = 2minutes
simulation.stop_time = 365days

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

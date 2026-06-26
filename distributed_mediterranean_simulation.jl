# # Distributed Mediterranean simulation with open boundary conditions at Gibraltar
#
# This is the distributed (MPI) counterpart of `mediterranean_simulation.jl`: the same high-resolution
# Copernicus bathymetry and GLORYS-fed Gibraltar open boundary conditions, partitioned across ranks.
# The reusable building blocks live in the `OceanMed` module.

# ## Initial setup with package imports

using MPI
using CUDA
MPI.Init()

using OceanMed
using OceanMed: MEDSEABathymetry, copernicus_z_faces,
                atlantic_sponge_forcings, atlantic_boundary_conditions

using Oceananigans
using Oceananigans.Units
using Oceananigans.Advection: FluxFormAdvection
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox

using Printf
using Dates

# ## Grid configuration for the Mediterranean Sea
#
# `synchronized_communication` is needed because vertical mixing does not work with pipelining.

@info "Initializing architecture"

arch = Distributed(GPU(), partition = Partition(4, 2), synchronized_communication = true)

const λ₁, λ₂ = (-8.6, 42) # domain in longitude
const φ₁, φ₂ = (  30, 48) # domain in latitude

const resolution = 1/24 # degrees, matching the MEDSEA bathymetry native grid

Nx = round(Int, (λ₂ - λ₁) / resolution)
Ny = round(Int, (φ₂ - φ₁) / resolution)

r_faces = copernicus_z_faces() # 280 levels reconstructed from data/zc_copernicus.nc
Nz = length(r_faces) - 1
z_faces = MutableVerticalDiscretization(r_faces)

@info "Building grid variable"

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude  = (φ₁, φ₂),
                             longitude = (λ₁, λ₂),
                             z = z_faces,
                             halo = (7, 7, 7))

# ### High-resolution Mediterranean bathymetry

bathymetry = Metadatum(:bottom_height; dataset = MEDSEABathymetry(), dir = "./data")

bottom_height = regrid_bathymetry(grid, bathymetry;
                                  minimum_depth = 5,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

@info arch.local_rank "We managed to create a distributed immersed grid"

# ## Open boundary conditions and sponge at Gibraltar (GLORYS)

start_date = DateTime(1993, 1, 1)
end_date   = DateTime(1994, 1, 1)

dataset = GLORYSMonthly()

bbox = BoundingBox(longitude = (λ₁ - 2, λ₂ + 2),
                   latitude  = (φ₁ - 2, φ₂ + 2))

dir = "./data"

u_meta = Metadata(:u_velocity;   dataset, dir, bounding_box = bbox, start_date, end_date)
v_meta = Metadata(:v_velocity;   dataset, dir, bounding_box = bbox, start_date, end_date)
T_meta = Metadata(:temperature;  dataset, dir, bounding_box = bbox, start_date, end_date)
S_meta = Metadata(:salinity;     dataset, dir, bounding_box = bbox, start_date, end_date)
η_meta = Metadata(:free_surface; dataset, dir, bounding_box = bbox, start_date, end_date)

u_glorys = FieldTimeSeries(u_meta, grid; time_indices_in_memory = 2, inpainting = 100)
v_glorys = FieldTimeSeries(v_meta, grid; time_indices_in_memory = 2, inpainting = 100)
T_glorys = FieldTimeSeries(T_meta, grid; time_indices_in_memory = 2, inpainting = 100)
S_glorys = FieldTimeSeries(S_meta, grid; time_indices_in_memory = 2, inpainting = 100)
η_glorys = FieldTimeSeries(η_meta, grid; time_indices_in_memory = 2, inpainting = 100)

forcing = atlantic_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta;
                                    west_longitude = λ₁,
                                    sponge_width = 2.0,
                                    tracer_rate = 1 / 1days,
                                    velocity_rate = 1 / 20minutes)

boundary_conditions = atlantic_boundary_conditions(grid, u_glorys, v_glorys, T_glorys, S_glorys, η_glorys;
                                                    inflow_timescale = 1days,
                                                    outflow_timescale = Inf)

# ## Constructing the simulation

momentum_advection = WENOVectorInvariant()
high_order = WENO(order = 7)
low_order  = Centered()
tracer_advection = FluxFormAdvection(high_order, high_order, low_order)

free_surface = SplitExplicitFreeSurface(grid; substeps = 80)

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         forcing,
                         boundary_conditions)

@info arch.local_rank "The med model has been built!!"

# Initialize temperature and salinity from GLORYS data.
set!(ocean.model, T = Metadatum(:temperature; date = start_date, dataset = GLORYSMonthly()),
                  S = Metadatum(:salinity;    date = start_date, dataset = GLORYSMonthly()))

# ## Atmospheric forcing

atmosphere = JRA55PrescribedAtmosphere(arch)

radiation = Radiation()

# The coupled model (no sea ice).
coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

# ## The coupled simulation

Δt = 5
stop_time = 10days
simulation = Simulation(coupled_model; Δt, stop_time)

wall_time = Ref(time_ns())

function progress(sim)
    u, v, w = sim.model.ocean.model.velocities
    T, S = sim.model.ocean.model.tracers
    rank = arch.local_rank

    step_time = (time_ns() - wall_time[]) * 1e-9
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

# Surface fields written every 24 hours.
simulation.output_writers[:surface_fields] = JLD2Writer(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                        schedule = TimeInterval(24hours),
                                                        indices = (:, :, grid.Nz),
                                                        overwrite_existing = true,
                                                        filename = "distributed_med_surface_fields")

ocean.output_writers[:checkpointer] = Checkpointer(ocean.model,
                                                   schedule = IterationInterval(86400),
                                                   overwrite_existing = true,
                                                   prefix = "distributed_mediterranean_rank$(arch.local_rank)")

# ## Running the simulation

run!(simulation)

simulation.Δt = 2minutes
simulation.stop_time = 365days

run!(simulation)

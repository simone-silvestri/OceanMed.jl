# # Mediterranean simulation with open boundary conditions at Gibraltar
#
# This example sets up and runs a high-resolution ocean simulation for the Mediterranean Sea using
# Oceananigans and NumericalEarth. It uses the high-resolution Copernicus Mediterranean bathymetry
# ([`MEDSEABathymetry`](@ref)) and open boundary conditions (OBCs) at the Strait of Gibraltar fed by
# GLORYS reanalysis data. The reusable building blocks live in the `OceanMed` module.

# ## Initial setup with package imports

using OceanMed
using OceanMed: MEDSEABathymetry, copernicus_z_faces,
                gibraltar_sponge_forcings, gibraltar_boundary_conditions

using Oceananigans
using Oceananigans.Units
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox

using CairoMakie
using Printf
using Dates

# ## Grid configuration for the Mediterranean Sea
#
# The grid spans the Mediterranean in longitude (λ₁, λ₂) and latitude (φ₁, φ₂) at 1/24ᵗʰ of a degree —
# the native resolution of the MEDSEA bathymetry, the finest topography available for this domain — with
# a stretched vertical grid reconstructed from the Copernicus levels (280 layers).

arch = GPU()

const λ₁, λ₂ = (-8.6, 36.5) # domain in longitude
const φ₁, φ₂ = (  30, 48)   # domain in latitude

const resolution = 1/24 # degrees, matching the MEDSEA bathymetry native grid

Nx = round(Int, (λ₂ - λ₁) / resolution)
Ny = round(Int, (φ₂ - φ₁) / resolution)

r_faces = copernicus_z_faces() # 280 levels reconstructed from data/zc_copernicus.nc
Nz = length(r_faces) - 1
z_faces = MutableVerticalDiscretization(r_faces)

# To run on distributed architectures (for example 4 ranks in x and 4 in y):
# arch = Distributed(arch, partition = Partition(x = 4, y = 4))

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             latitude  = (φ₁, φ₂),
                             longitude = (λ₁, λ₂),
                             z = z_faces,
                             halo = (7, 7, 7))

# ### High-resolution Mediterranean bathymetry
#
# `MEDSEABathymetry` registers the static bottom topography of the Copernicus product
# `MEDSEA_ANALYSISFORECAST_PHY_006_013` (~4.2 km) as a NumericalEarth dataset, downloaded through the
# Copernicus Marine "Data Access" service. Downloading requires the `COPERNICUS_USERNAME` and
# `COPERNICUS_PASSWORD` environment variables. Replace the dataset with `ETOPO2022()` for the global
# default bathymetry.

bathymetry = Metadatum(:bottom_height; dataset = MEDSEABathymetry(), dir = "./data")

bottom_height = regrid_bathymetry(grid, bathymetry;
                                  minimum_depth = 5,
                                  interpolation_passes = 5,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map = true)

# ## Open boundary conditions at Gibraltar using GLORYS
#
# Instead of restoring to global data near Gibraltar, we impose open boundary conditions on the western
# boundary (just outside the Strait at λ₁ = -8.6°), fed by the GLORYS reanalysis: a baroclinic `Radiation`
# condition and a barotropic `Flather` condition, combined with a sponge layer (see the `OceanMed` helpers).

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

# Load GLORYS data as FieldTimeSeries interpolated onto the simulation grid. Inpainting fills the
# GLORYS land/under-bathymetry points so the western boundary column has valid data at depth.
u_glorys = FieldTimeSeries(u_meta, grid; time_indices_in_memory = 2, inpainting = 100)
v_glorys = FieldTimeSeries(v_meta, grid; time_indices_in_memory = 2, inpainting = 100)
T_glorys = FieldTimeSeries(T_meta, grid; time_indices_in_memory = 2, inpainting = 100)
S_glorys = FieldTimeSeries(S_meta, grid; time_indices_in_memory = 2, inpainting = 100)
η_glorys = FieldTimeSeries(η_meta, grid; time_indices_in_memory = 2, inpainting = 100)

# ## Forcing and boundary conditions
#
# The western (Gibraltar) boundary is the only open edge. Its baroclinic velocities and tracers get an
# Orlanski `Radiation` open boundary, while the barotropic transport gets a `Flather` condition — both
# fed by GLORYS. A `DatasetRestoring` sponge just inside the boundary complements them by relaxing the
# near-boundary interior towards GLORYS (see the `OceanMed` helpers).

forcing = gibraltar_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta;
                                    west_longitude = λ₁,
                                    sponge_width = 2.0,             # degrees (~200 km)
                                    tracer_rate = 1 / 1days,
                                    velocity_rate = 1 / 20minutes)

boundary_conditions = gibraltar_boundary_conditions(grid, u_glorys, v_glorys, T_glorys, S_glorys, η_glorys;
                                                    inflow_timescale = 1days,   # relax to GLORYS on inflow
                                                    outflow_timescale = Inf)    # let outgoing waves radiate freely

# ## Constructing the simulation
#
# An ocean simulation that evolves temperature (:T) and salinity (:S) with the open western boundary.
# The split-explicit free surface carries the barotropic mode that the `Flather` boundary acts on.

momentum_advection = WENOVectorInvariant()
tracer_advection   = WENO(order = 7)
free_surface       = SplitExplicitFreeSurface(grid; substeps = 80)

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         free_surface,
                         forcing,
                         boundary_conditions)

# Initialize temperature and salinity from GLORYS data.
set!(ocean.model, T = Metadatum(:temperature; date = start_date, dataset = GLORYSMonthly()),
                  S = Metadatum(:salinity;    date = start_date, dataset = GLORYSMonthly()))

# ## Atmospheric forcing
#
# We force the model with the JRA55-do dataset (surface heat fluxes and wind stress). Only 10 time
# instances are held in memory at a time and updated as the model progresses.

atmosphere = JRA55PrescribedAtmosphere(arch; backend = JRA55NetCDFBackend(10),
                                       include_rivers_and_icebergs = true, dir = "./data")

radiation = Radiation()

# The coupled model (no sea ice).
coupled_model = OceanSeaIceModel(ocean; atmosphere, radiation)

# ## The coupled simulation

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

# Surface fields written every 24 hours.
simulation.output_writers[:surface_fields] = JLD2Writer(ocean.model, merge(ocean.model.tracers, ocean.model.velocities),
                                                        schedule = TimeInterval(24hours),
                                                        indices = (:, :, grid.Nz),
                                                        overwrite_existing = true,
                                                        filename = "med_surface_fields.jld2")

ocean.output_writers[:checkpointer] = Checkpointer(ocean.model,
                                                   schedule = IterationInterval(86400),
                                                   overwrite_existing = true,
                                                   prefix = "mediterranean")

# ## Running the simulation
#
# A short spin-up at a small time step, then a year-long run at a larger time step.

run!(simulation)

simulation.Δt = 2minutes
simulation.stop_time = 365days

run!(simulation)

# ## Recording a video
#
# Read back the surface fields and record a video of (1) zonal velocity, (2) meridional velocity,
# (3) temperature, and (4) salinity.

u_series = FieldTimeSeries("med_surface_fields.jld2", "u"; backend = OnDisk())
v_series = FieldTimeSeries("med_surface_fields.jld2", "v"; backend = OnDisk())
T_series = FieldTimeSeries("med_surface_fields.jld2", "T"; backend = OnDisk())
S_series = FieldTimeSeries("med_surface_fields.jld2", "S"; backend = OnDisk())
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

# # Mediterranean simulation with open boundary conditions at Gibraltar
#
# This example sets up and runs a high-resolution ocean simulation for the Mediterranean Sea
# using the Oceananigans and NumericalEarth packages, with open boundary conditions (OBCs) at
# the Strait of Gibraltar using GLORYS reanalysis data.

# ## Initial Setup with Package Imports

using Pkg
using CairoMakie
using Oceananigans
using Oceananigans.Grids
using Oceananigans: architecture
using Oceananigans.BoundaryConditions: RadiationBoundaryCondition
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, download_dataset
using NumericalEarth.Oceans: u_quadratic_bottom_drag, v_quadratic_bottom_drag,
                             u_immersed_bottom_drag, v_immersed_bottom_drag
using Oceananigans.Units
using Oceananigans.Grids: xnode
using Printf
using NCDatasets
using Dates

include("vertical_diffusivity.jl")

# ## Grid Configuration for the Mediterranean Sea
#
# The script defines a high-resolution grid to represent the Mediterranean Sea, specifying the domain in terms of longitude (λ₁, λ₂),
# latitude (φ₁, φ₂), and a stretched vertical grid to capture the depth variation (`z_faces`).
# The grid resolution is set to approximately 1/30th of a degree.
# This section demonstrates the use of the LatitudeLongitudeGrid function to create a grid that matches the
# Mediterranean's geographical and bathymetric features.

arch = GPU()

const λ₁, λ₂  = (-8.6, 42) # domain in longitude
const φ₁, φ₂  = (  30, 48) # domain in latitude

Nx = 30 * ceil(Int, λ₂ - λ₁) # ~1/30th of a degree resolution
Ny = 30 * ceil(Int, φ₂ - φ₁) # ~1/30th of a degree resolution
# Build z-faces by first reconstructing the 141 Copernicus faces from
# the 140 depth centers, then splitting each layer in half → 280 levels.
ds = Dataset("data/zc_copernicus.nc")
r_centers = reverse(ds["depth"][:])
Nz_copernicus = length(r_centers)
coarse_faces = zeros(Nz_copernicus + 1)
for k in Nz_copernicus:-1:1
    coarse_faces[k] = coarse_faces[k+1] - (r_centers[k] + coarse_faces[k+1]) * 2
end

# Insert a midpoint between each pair of coarse faces → 281 faces, 280 layers
Nz = 2 * Nz_copernicus
r_faces = zeros(Nz + 1)
for k in 1:Nz_copernicus
    r_faces[2k - 1] = coarse_faces[k]
    r_faces[2k]     = (coarse_faces[k] + coarse_faces[k+1]) / 2
end
r_faces[end] = coarse_faces[end]

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

# ## Open Boundary Conditions at Gibraltar using GLORYS
#
# Instead of restoring to ECCO data near Gibraltar, we use open boundary conditions (OBCs)
# on the western boundary of the domain (just outside the Strait of Gibraltar at λ₁ = -8.6°).
# The external data for the OBCs comes from the GLORYS reanalysis dataset.
# We use the Orlanski radiation scheme with adaptive nudging (Marchesiello et al. 2001).

start_date = DateTime(1993, 1, 1)
end_date   = DateTime(1994, 1, 1)

dataset = GLORYSMonthly()

bbox = BoundingBox(longitude = (λ₁ - 2, λ₂ + 2),
                   latitude  = (φ₁ - 2, φ₂ + 2))

dir = "./data"

u_meta = Metadata(:u_velocity;  dataset, dir, bounding_box=bbox, start_date, end_date)
v_meta = Metadata(:v_velocity;  dataset, dir, bounding_box=bbox, start_date, end_date)
T_meta = Metadata(:temperature; dataset, dir, bounding_box=bbox, start_date, end_date)
S_meta = Metadata(:salinity;    dataset, dir, bounding_box=bbox, start_date, end_date)

# Load GLORYS data as FieldTimeSeries interpolated onto the simulation grid
u_glorys = FieldTimeSeries(u_meta, grid; time_indices_in_memory=2)
v_glorys = FieldTimeSeries(v_meta, grid; time_indices_in_memory=2)
T_glorys = FieldTimeSeries(T_meta, grid; time_indices_in_memory=2)
S_glorys = FieldTimeSeries(S_meta, grid; time_indices_in_memory=2)

# Discrete boundary condition function: extracts the GLORYS value at the western boundary
@inline function west_boundary_value(j, k, grid, clock, model_fields, fts)
    time = Time(clock.time)
    return @inbounds fts[1, j, k, time]
end

# ## Sponge layers
#
# Radiation OBCs are imperfect — Orlanski diagnoses a single phase speed but
# internal waves have multiple baroclinic modes. A sponge layer near the
# western boundary damps waves before they reach the boundary, reducing reflections.
# The sponge uses a cos²(πd/2W) profile that ramps smoothly from 1 at the
# boundary to 0 at the interior edge of the sponge region.

@inline function sponge_mask(λ, p)
    d = λ - p.λ₁  # distance from west boundary (positive going east)
    return ifelse(d < p.W, cospi(d / (2 * p.W))^2, zero(d))
end

@inline function u_sponge_forcing(i, j, k, grid, clock, fields, p)
    λ = xnode(i, grid, Face())
    μ = sponge_mask(λ, p)
    time = Time(clock.time)
    u_ext = @inbounds p.fts[i, j, k, time]
    return -p.rate * μ * (@inbounds fields.u[i, j, k] - u_ext)
end

@inline function v_sponge_forcing(i, j, k, grid, clock, fields, p)
    λ = xnode(i, grid, Center())
    μ = sponge_mask(λ, p)
    time = Time(clock.time)
    v_ext = @inbounds p.fts[i, j, k, time]
    return -p.rate * μ * (@inbounds fields.v[i, j, k] - v_ext)
end

@inline function T_sponge_forcing(i, j, k, grid, clock, fields, p)
    λ = xnode(i, grid, Center())
    μ = sponge_mask(λ, p)
    time = Time(clock.time)
    T_ext = @inbounds p.fts[i, j, k, time]
    return -p.rate * μ * (@inbounds fields.T[i, j, k] - T_ext)
end

@inline function S_sponge_forcing(i, j, k, grid, clock, fields, p)
    λ = xnode(i, grid, Center())
    μ = sponge_mask(λ, p)
    time = Time(clock.time)
    S_ext = @inbounds p.fts[i, j, k, time]
    return -p.rate * μ * (@inbounds fields.S[i, j, k] - S_ext)
end

sponge_width = 2.0  # degrees (~200 km)
sponge_rate  = 1 / 1hour

u_sponge = Forcing(u_sponge_forcing, discrete_form=true, parameters=(; λ₁, W=sponge_width, rate=sponge_rate, fts=u_glorys))
v_sponge = Forcing(v_sponge_forcing, discrete_form=true, parameters=(; λ₁, W=sponge_width, rate=sponge_rate, fts=v_glorys))
T_sponge = Forcing(T_sponge_forcing, discrete_form=true, parameters=(; λ₁, W=sponge_width, rate=sponge_rate, fts=T_glorys))
S_sponge = Forcing(S_sponge_forcing, discrete_form=true, parameters=(; λ₁, W=sponge_width, rate=sponge_rate, fts=S_glorys))

# ## Constructing boundary conditions
#
# We need to construct full FieldBoundaryConditions that include:
# - Top: surface flux fields (filled by the coupled model)
# - Bottom: quadratic bottom drag
# - Immersed: immersed boundary drag
# - West: Radiation OBCs with GLORYS external data

# Surface flux fields (filled by OceanSeaIceModel during coupling)
zonal_momentum_flux      = Field{Face, Center, Nothing}(grid)
meridional_momentum_flux = Field{Center, Face, Nothing}(grid)
temperature_flux         = Field{Center, Center, Nothing}(grid)
salinity_flux            = Field{Center, Center, Nothing}(grid)

# Bottom drag parameters
drag_parameters = (μ = 0.003, Uᴮ = 0.05)

# Radiation OBC relaxation timescales
outflow_timescale = Inf    # let outgoing waves pass freely
inflow_timescale  = 300.0  # nudge on inflow (5 minutes, Marchesiello et al. 2001)

u_bcs = FieldBoundaryConditions(
    top      = FluxBoundaryCondition(zonal_momentum_flux),
    bottom   = FluxBoundaryCondition(u_quadratic_bottom_drag, discrete_form=true, parameters=drag_parameters),
    immersed = ImmersedBoundaryCondition(bottom=FluxBoundaryCondition(u_immersed_bottom_drag, discrete_form=true, parameters=drag_parameters)),
    west     = RadiationBoundaryCondition(west_boundary_value; discrete_form=true, parameters=u_glorys,
                                          outflow_relaxation_timescale=outflow_timescale, inflow_relaxation_timescale=inflow_timescale))

v_bcs = FieldBoundaryConditions(
    top      = FluxBoundaryCondition(meridional_momentum_flux),
    bottom   = FluxBoundaryCondition(v_quadratic_bottom_drag, discrete_form=true, parameters=drag_parameters),
    immersed = ImmersedBoundaryCondition(bottom=FluxBoundaryCondition(v_immersed_bottom_drag, discrete_form=true, parameters=drag_parameters)),
    west     = RadiationBoundaryCondition(west_boundary_value; discrete_form=true, parameters=v_glorys,
                                          outflow_relaxation_timescale=outflow_timescale, inflow_relaxation_timescale=inflow_timescale))

T_bcs = FieldBoundaryConditions(
    top  = FluxBoundaryCondition(temperature_flux),
    west = RadiationBoundaryCondition(west_boundary_value; discrete_form=true, parameters=T_glorys,
                                      outflow_relaxation_timescale=outflow_timescale, inflow_relaxation_timescale=inflow_timescale))

S_bcs = FieldBoundaryConditions(
    top  = FluxBoundaryCondition(salinity_flux),
    west = RadiationBoundaryCondition(west_boundary_value; discrete_form=true, parameters=S_glorys,
                                      outflow_relaxation_timescale=outflow_timescale, inflow_relaxation_timescale=inflow_timescale))

# Constructing the Simulation
#
# We construct an ocean simulation that evolves two tracers, temperature (:T), salinity (:S)
# with OBCs on the western boundary.

momentum_advection = WENOVectorInvariant()
tracer_advection = WENO(order = 7)
timestepper = :SplitRungeKutta3

ocean = ocean_simulation(grid;
                         momentum_advection,
                         tracer_advection,
                         forcing = (u=u_sponge, v=v_sponge, T=T_sponge, S=S_sponge),
                         boundary_conditions = (u=u_bcs, v=v_bcs, T=T_bcs, S=S_bcs))

# Initializing the model
#
# Initialize temperature and salinity from GLORYS data.

set!(ocean.model, T=Metadatum(:temperature; date=start_date, dataset=GLORYSMonthly()),
                  S=Metadatum(:salinity;    date=start_date, dataset=GLORYSMonthly()))

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

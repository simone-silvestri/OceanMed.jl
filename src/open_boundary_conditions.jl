module OpenBoundaryConditions

export atlantic_boundary_conditions, atlantic_sponge_forcings, AtlanticSpongeMask

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: node
using Oceananigans.Operators: Δzᶠᶜᶜ, Δzᶜᶠᶜ
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.BoundaryConditions: Radiation, FlatherBoundaryCondition,
                                       NormalFlowBoundaryCondition, ValueBoundaryCondition

using NumericalEarth: DatasetRestoring
import NumericalEarth: stateindex

#####
##### Baroclinic open boundary: external value read from a GLORYS FieldTimeSeries
#####

# Evaluate the external field at an open-boundary cell (west column i = 1, north row j = Ny, south row
# j = 1) and the current clock time. West is an x-boundary indexed by (j, k); north/south are
# y-boundaries indexed by (i, k).
@inline west_boundary_value( j, k, grid, clock, fields, φ) = @inbounds φ[1,       j, k, Time(clock.time)]
@inline north_boundary_value(i, k, grid, clock, fields, φ) = @inbounds φ[i, grid.Ny, k, Time(clock.time)]
@inline south_boundary_value(i, k, grid, clock, fields, φ) = @inbounds φ[i,       1, k, Time(clock.time)]

#####
##### Barotropic open boundary (Flather): external transport + free-surface elevation
#####

@inline wetcell(i, j, k, grid, ℓx, ℓy, ℓz) =
    !immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) & !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

# Wet-column integrals of the external velocities → external barotropic transports ∫u dz and ∫v dz.
@inline function zonal_transport(i, j, grid, u, t)
    U = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = wetcell(i, j, k, grid, Face(), Center(), Center())
        U += ifelse(wet, u[i, j, k, t] * Δzᶠᶜᶜ(i, j, k, grid), zero(U))
    end
    return U
end

@inline function meridional_transport(i, j, grid, v, t)
    V = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = wetcell(i, j, k, grid, Center(), Face(), Center())
        V += ifelse(wet, v[i, j, k, t] * Δzᶜᶠᶜ(i, j, k, grid), zero(V))
    end
    return V
end

@inline function west_barotropic_value(j, k, grid, clock, fields, p)
    t = Time(clock.time)
    U = zonal_transport(1, j, grid, p.u, t)
    return (U, @inbounds p.η[1, j, 1, t])
end

@inline function north_barotropic_value(i, k, grid, clock, fields, p)
    t = Time(clock.time)
    V = meridional_transport(i, grid.Ny, grid, p.v, t)
    return (V, @inbounds p.η[i, grid.Ny, 1, t])
end

@inline function south_barotropic_value(i, k, grid, clock, fields, p)
    t = Time(clock.time)
    V = meridional_transport(i, 1, grid, p.v, t)
    return (V, @inbounds p.η[i, 1, 1, t])
end

"""
    atlantic_boundary_conditions(grid, u_glorys, v_glorys, T_glorys, S_glorys, η_glorys;
                                  inflow_timescale = 1days, outflow_timescale = Inf)

Build the Atlantic open boundary conditions for the Mediterranean, fed by the GLORYS `FieldTimeSeries`
`u_glorys`, `v_glorys`, `T_glorys`, `S_glorys` (baroclinic fields) and `η_glorys` (free-surface
elevation). The western edge (Strait of Gibraltar) is open over its full extent; the northern and
southern edges are open over the Atlantic, where the bathymetry now reaches the domain boundary (land
cells along those edges are inactive, so an edge-wide condition is harmless there). Returns a named
tuple of `FieldBoundaryConditions` for `(u, v, T, S, U, V)`:

- baroclinic velocities and tracers get an Orlanski-type `Radiation` scheme with adaptive nudging
  (Marchesiello et al. 2001): the boundary follows a locally diagnosed phase speed on outflow
  (`outflow_timescale = Inf`) and relaxes towards GLORYS on inflow (`inflow_timescale`). The
  boundary-normal velocity (`u` at west, `v` at north/south) uses a `NormalFlowBoundaryCondition`; the
  tangential velocity and the tracers `T`, `S` use a `ValueBoundaryCondition`;
- the barotropic transports get a `FlatherBoundaryCondition` (Flather 1976) fed by the external
  wet-column transport and the GLORYS free surface, so the depth-mean Atlantic exchange can actually
  cross the boundary: `U` (∫u dz, ηᵂ) at west, `V` (∫v dz, ηᴺ/ηˢ) at north/south.

Pass the result as `boundary_conditions = (; u, v, T, S, U, V)` to `ocean_simulation`, which merges
them with its default surface-flux and bottom-drag conditions. Use the matching `U`/`V` boundaries
with a `SplitExplicitFreeSurface`.
"""
function atlantic_boundary_conditions(grid, u_glorys, v_glorys, T_glorys, S_glorys, η_glorys;
                                       inflow_timescale = 1days, outflow_timescale = Inf)

    radiation() = Radiation(; inflow_timescale, outflow_timescale)

    u_bcs = FieldBoundaryConditions(west  = NormalFlowBoundaryCondition(west_boundary_value;  discrete_form = true, parameters = u_glorys, scheme = radiation()),
                                    north = ValueBoundaryCondition(north_boundary_value;      discrete_form = true, parameters = u_glorys, scheme = radiation()),
                                    south = ValueBoundaryCondition(south_boundary_value;      discrete_form = true, parameters = u_glorys, scheme = radiation()))

    v_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(west_boundary_value;        discrete_form = true, parameters = v_glorys, scheme = radiation()),
                                    north = NormalFlowBoundaryCondition(north_boundary_value;  discrete_form = true, parameters = v_glorys, scheme = radiation()),
                                    south = NormalFlowBoundaryCondition(south_boundary_value;  discrete_form = true, parameters = v_glorys, scheme = radiation()))

    T_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(west_boundary_value;  discrete_form = true, parameters = T_glorys, scheme = radiation()),
                                    north = ValueBoundaryCondition(north_boundary_value; discrete_form = true, parameters = T_glorys, scheme = radiation()),
                                    south = ValueBoundaryCondition(south_boundary_value; discrete_form = true, parameters = T_glorys, scheme = radiation()))

    S_bcs = FieldBoundaryConditions(west  = ValueBoundaryCondition(west_boundary_value;  discrete_form = true, parameters = S_glorys, scheme = radiation()),
                                    north = ValueBoundaryCondition(north_boundary_value; discrete_form = true, parameters = S_glorys, scheme = radiation()),
                                    south = ValueBoundaryCondition(south_boundary_value; discrete_form = true, parameters = S_glorys, scheme = radiation()))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
                                    west = FlatherBoundaryCondition(west_barotropic_value; discrete_form = true, parameters = (u = u_glorys, η = η_glorys)))

    V_bcs = FieldBoundaryConditions(grid, (Center(), Face(), nothing);
                                    north = FlatherBoundaryCondition(north_barotropic_value; discrete_form = true, parameters = (v = v_glorys, η = η_glorys)),
                                    south = FlatherBoundaryCondition(south_barotropic_value; discrete_form = true, parameters = (v = v_glorys, η = η_glorys)))

    return (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs, U = U_bcs, V = V_bcs)
end

#####
##### Sponge layer just inside the western boundary
#####

"""
    AtlanticSpongeMask(west_longitude, south_latitude, north_latitude, width, taper_longitude, taper_width)

Callable restoring mask that covers the Atlantic side of the domain along its three open edges. It is
the maximum of three thin Gaussian sponges, each decaying inward from a boundary over `width` degrees:

- a **western** sponge `exp(-(λ - west_longitude)² / 2 width²)` decaying eastward from `west_longitude`,
  over the whole latitude range;
- a **northern** sponge `exp(-(φ - north_latitude)² / 2 width²)` decaying southward from the northern
  edge `north_latitude`;
- a **southern** sponge decaying northward from the southern edge `south_latitude`.

The northern and southern sponges are multiplied by a longitudinal taper that is `1` over the Atlantic
(`λ ≤ taper_longitude`) and decays to zero to the east as `exp(-(λ - taper_longitude)² / 2 taper_width²)`,
so they stay confined to the open Atlantic edges and never reach the Mediterranean interior.

Used as the `mask` of a `DatasetRestoring` forcing. Being a small isbits object, it is safe to pass
to GPU kernels.
"""
struct AtlanticSpongeMask{FT}
    west_longitude  :: FT
    south_latitude  :: FT
    north_latitude  :: FT
    width           :: FT
    taper_longitude :: FT
    taper_width     :: FT
end

AtlanticSpongeMask(west_longitude, south_latitude, north_latitude, width, taper_longitude, taper_width) =
    AtlanticSpongeMask(promote(west_longitude, south_latitude, north_latitude, width, taper_longitude, taper_width)...)

@inline function (mask::AtlanticSpongeMask)(λ, φ, z, t)
    west = exp(-(λ - mask.west_longitude)^2 / (2 * mask.width^2))

    east  = max(λ - mask.taper_longitude, zero(λ))
    taper = exp(-east^2 / (2 * mask.taper_width^2))

    north = exp(-(φ - mask.north_latitude)^2 / (2 * mask.width^2)) * taper
    south = exp(-(φ - mask.south_latitude)^2 / (2 * mask.width^2)) * taper

    return max(west, north, south)
end

@inline function stateindex(mask::AtlanticSpongeMask, i, j, k, grid, time, loc)
    LX, LY, LZ = loc
    λ, φ, z = node(i, j, k, grid, LX(), LY(), LZ())
    return mask(λ, φ, z, time)
end

"""
    atlantic_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta;
                              west_longitude, south_latitude, north_latitude,
                              sponge_width = 2.0, taper_longitude = 0.0, taper_width = 1.0,
                              tracer_rate = 1 / 1days, velocity_rate = 1 / 20minutes, inpainting = 100)

Build the `(T, S, u, v)` named tuple of `DatasetRestoring` forcings that relax the prognostic fields
towards the GLORYS data (`T_meta`, `S_meta`, `u_meta`, `v_meta`) within an `AtlanticSpongeMask` that
sponges the western boundary at `west_longitude` (over all latitudes) and the northern and southern
edges at `north_latitude`/`south_latitude`, each a thin Gaussian decaying inward and the meridional
ones tapered to zero east of `taper_longitude` so they stay on the open Atlantic edges. `sponge_width`
sets the Gaussian decay scale away from each edge and `taper_width` the scale of the longitudinal
taper (all in degrees). Tracers relax on the gentle `tracer_rate`; the velocities use
the stronger `velocity_rate`, keeping the near-boundary interior consistent with the prescribed
open-boundary inflow. The sponge complements the open boundary by damping oblique waves and slow
drift that radiation alone cannot absorb.
"""
function atlantic_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta;
                                  west_longitude,
                                  south_latitude,
                                  north_latitude,
                                  sponge_width = 2.0,
                                  taper_longitude = 0.0,
                                  taper_width = 1.0,
                                  tracer_rate = 1 / 1days,
                                  velocity_rate = 1 / 20minutes,
                                  inpainting = 100)

    mask = AtlanticSpongeMask(west_longitude, south_latitude, north_latitude,
                              sponge_width, taper_longitude, taper_width)

    T = DatasetRestoring(T_meta, grid; rate = tracer_rate,   mask, inpainting)
    S = DatasetRestoring(S_meta, grid; rate = tracer_rate,   mask, inpainting)
    u = DatasetRestoring(u_meta, grid; rate = velocity_rate, mask, inpainting)
    v = DatasetRestoring(v_meta, grid; rate = velocity_rate, mask, inpainting)

    return (; T, S, u, v)
end

end # module OpenBoundaryConditions

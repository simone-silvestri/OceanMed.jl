module OpenBoundaryConditions

export gibraltar_boundary_conditions, gibraltar_sponge_forcings, WesternSpongeMask

using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators: Δzᶠᶜᶜ
using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.BoundaryConditions: Radiation, FlatherBoundaryCondition,
                                       NormalFlowBoundaryCondition, ValueBoundaryCondition

using NumericalEarth: DatasetRestoring

#####
##### Baroclinic open boundary: external value read from a GLORYS FieldTimeSeries
#####

# Evaluate the external field at the western boundary column (i = 1) and the current clock time.
@inline west_boundary_value(j, k, grid, clock, fields, φ) = @inbounds φ[1, j, k, Time(clock.time)]

#####
##### Barotropic open boundary (Flather): external transport + free-surface elevation
#####

@inline wetcell(i, j, k, grid, ℓx, ℓy, ℓz) =
    !immersed_peripheral_node(i, j, k, grid, ℓx, ℓy, ℓz) & !immersed_inactive_node(i, j, k, grid, ℓx, ℓy, ℓz)

# Wet-column integral of the external zonal velocity → external barotropic transport ∫u dz.
@inline function vertical_transport(i, j, grid, u, t)
    U = zero(eltype(grid))
    @inbounds for k in 1:grid.Nz
        wet = wetcell(i, j, k, grid, Face(), Center(), Center())
        U += ifelse(wet, u[i, j, k, t] * Δzᶠᶜᶜ(i, j, k, grid), zero(U))
    end
    return U
end

@inline function west_barotropic_value(j, k, grid, clock, fields, p)
    t = Time(clock.time)
    U = vertical_transport(1, j, grid, p.u, t)
    return (U, @inbounds p.η[1, j, 1, t])
end

"""
    gibraltar_boundary_conditions(grid, u_glorys, v_glorys, T_glorys, S_glorys, η_glorys;
                                  inflow_timescale = 1days, outflow_timescale = Inf)

Build the western (Strait of Gibraltar) open boundary conditions for the Mediterranean, fed by the
GLORYS `FieldTimeSeries` `u_glorys`, `v_glorys`, `T_glorys`, `S_glorys` (baroclinic fields) and
`η_glorys` (free-surface elevation). Returns a named tuple of `FieldBoundaryConditions` for
`(u, v, T, S, U)`:

- baroclinic velocities and tracers get an Orlanski-type `Radiation` scheme with adaptive nudging
  (Marchesiello et al. 2001): the boundary follows a locally diagnosed phase speed on outflow
  (`outflow_timescale = Inf`) and relaxes towards GLORYS on inflow (`inflow_timescale`). The
  boundary-normal velocity `u` uses a `NormalFlowBoundaryCondition`; the tangential velocity `v`
  and the tracers `T`, `S` use a `ValueBoundaryCondition`;
- the barotropic transport `U` gets a `FlatherBoundaryCondition` (Flather 1976) fed by the external
  wet-column transport `∫u dz` and the GLORYS free surface, so the depth-mean Atlantic inflow can
  actually cross the boundary.

Only the western boundary is open; the remaining edges stay closed (land). Pass the result as
`boundary_conditions = (; u, v, T, S, U)` to `ocean_simulation`, which merges them with its default
surface-flux and bottom-drag conditions. Use the matching `U` boundary with a
`SplitExplicitFreeSurface`.
"""
function gibraltar_boundary_conditions(grid, u_glorys, v_glorys, T_glorys, S_glorys, η_glorys;
                                       inflow_timescale = 1days, outflow_timescale = Inf)

    radiation() = Radiation(; inflow_timescale, outflow_timescale)

    u_bcs = FieldBoundaryConditions(west = NormalFlowBoundaryCondition(west_boundary_value; discrete_form = true, parameters = u_glorys, scheme = radiation()))
    v_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(west_boundary_value;      discrete_form = true, parameters = v_glorys, scheme = radiation()))
    T_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(west_boundary_value;      discrete_form = true, parameters = T_glorys, scheme = radiation()))
    S_bcs = FieldBoundaryConditions(west = ValueBoundaryCondition(west_boundary_value;      discrete_form = true, parameters = S_glorys, scheme = radiation()))

    U_bcs = FieldBoundaryConditions(grid, (Face(), Center(), nothing);
        west = FlatherBoundaryCondition(west_barotropic_value; discrete_form = true, parameters = (u = u_glorys, η = η_glorys)))

    return (; u = u_bcs, v = v_bcs, T = T_bcs, S = S_bcs, U = U_bcs)
end

#####
##### Sponge layer just inside the western boundary
#####

"""
    WesternSpongeMask(west_longitude, width)

Callable restoring mask that decays as a Gaussian `exp(-(λ - west_longitude)² / 2 width²)` away from
the western boundary at `west_longitude` (both in degrees). Used as the `mask` of a `DatasetRestoring`
forcing. Being a small isbits object, it is safe to pass to GPU kernels.
"""
struct WesternSpongeMask{FT}
    west_longitude :: FT
    width :: FT
end

WesternSpongeMask(west_longitude, width) = WesternSpongeMask(promote(west_longitude, width)...)

@inline (mask::WesternSpongeMask)(λ, φ, z, t) = exp(-(λ - mask.west_longitude)^2 / (2 * mask.width^2))

"""
    gibraltar_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta;
                              west_longitude, sponge_width = 2.0,
                              tracer_rate = 1 / 1days, velocity_rate = 1 / 20minutes, inpainting = 100)

Build the `(T, S, u, v)` named tuple of `DatasetRestoring` forcings that relax the prognostic fields
towards the GLORYS data (`T_meta`, `S_meta`, `u_meta`, `v_meta`) within a Gaussian sponge of width
`sponge_width` degrees just inside the western boundary at `west_longitude`. Tracers relax on the
gentle `tracer_rate`; the velocities use the stronger `velocity_rate`, keeping the near-boundary
interior consistent with the prescribed open-boundary inflow. The sponge complements the open
boundary by damping oblique waves and slow drift that radiation alone cannot absorb.
"""
function gibraltar_sponge_forcings(grid, T_meta, S_meta, u_meta, v_meta;
                                   west_longitude, sponge_width = 2.0,
                                   tracer_rate = 1 / 1days, velocity_rate = 1 / 20minutes, inpainting = 100)

    mask = WesternSpongeMask(west_longitude, sponge_width)

    T = DatasetRestoring(T_meta, grid; rate = tracer_rate,   mask, inpainting)
    S = DatasetRestoring(S_meta, grid; rate = tracer_rate,   mask, inpainting)
    u = DatasetRestoring(u_meta, grid; rate = velocity_rate, mask, inpainting)
    v = DatasetRestoring(v_meta, grid; rate = velocity_rate, mask, inpainting)

    return (; T, S, u, v)
end

end # module OpenBoundaryConditions

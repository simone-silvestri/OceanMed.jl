#####
##### Conservative regridding of the CERRA atmosphere onto the exchange grid, at coupling time.
#####
##### A CERRA `PrescribedAtmosphere` carries its `FieldTimeSeries` on a native Lambert
##### (`OrthogonalSphericalShellGrid`) grid. The default coupler interpolates the atmosphere onto the
##### exchange grid with fractional indices, which assume an invertible (lat–lon) source grid. For the
##### curvilinear CERRA grid we instead store a `ConservativeRegridding.Regridder` in the atmosphere
##### exchanger and apply it (a sparse mat-vec, GPU-capable) every timestep.
#####
##### The integration dispatches on the atmosphere's grid type, so it needs no edits to NumericalEarth.
#####

using Oceananigans.Grids: OrthogonalSphericalShellGrid
using Oceananigans.Architectures: on_architecture, architecture
using ConservativeRegridding

using NumericalEarth: PrescribedAtmosphere
using NumericalEarth.Oceans: forcing_barotropic_potential
using NumericalEarth.Radiations: PrescribedRadiation

import NumericalEarth.EarthSystemModels: interpolate_state!
import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, initialize!, materialize_correction

# A CERRA-style atmosphere is any PrescribedAtmosphere whose FieldTimeSeries live on a curvilinear grid.
const CurvilinearAtmosphere = PrescribedAtmosphere{<:Any, <:OrthogonalSphericalShellGrid}

# Holds the conservative regridder (native → exchange) plus a single reusable native time-buffer.
# Used for components whose fields are regridded directly (the radiation, below).
struct ConservativeAtmosphereRegridder{R, B}
    regridder :: R
    native_buffer :: B
end

# Time-interpolate `fts` onto the native buffer, then conservatively regrid it into `destination`.
function regrid_atmosphere_field!(destination, regridding::ConservativeAtmosphereRegridder, fts, time)
    buffer = regridding.native_buffer
    set!(buffer, fts[Time(time)])
    ConservativeRegridding.regrid!(destination, regridding.regridder, buffer)
    return destination
end

#####
##### Pointwise derivations (CERRA gives wind speed/direction and relative humidity, not u/v and q).
##### Applied on the native grid each timestep by the `interpolate_state!` kernel.
#####

@inline wind_u(speed, direction) = - speed * sind(direction)   # meteorological "from" direction
@inline wind_v(speed, direction) = - speed * cosd(direction)

# specific humidity from relative humidity (%), temperature (K), pressure (Pa); Tetens saturation.
@inline function specific_humidity(relative_humidity, temperature, pressure)
    Tc = temperature - 273.15
    saturation_pressure = 611.2 * exp(17.67 * Tc / (Tc + 243.5))
    vapor_pressure = (relative_humidity / 100) * saturation_pressure
    return 0.622 * vapor_pressure / (pressure - 0.378 * vapor_pressure)
end

#####
##### Atmosphere exchanger: derive `u, v, q, rain, snow` on the native grid, then regrid.
#####

struct CERRAAtmosphereRegridder{R, RAW, DER}
    regridder :: R
    raw       :: RAW  # (speed, direction, T, rh, p, precip, snowfall)
    derived   :: DER  # (u, v, q, rain) — snow is `snowfall` directly
end

function ComponentExchanger(atmosphere::CurvilinearAtmosphere, grid; correction = nothing)
    state = (; u   = Field{Center, Center, Nothing}(grid),
               v   = Field{Center, Center, Nothing}(grid),
               T   = Field{Center, Center, Nothing}(grid),
               p   = Field{Center, Center, Nothing}(grid),
               q   = Field{Center, Center, Nothing}(grid),
               Jʳⁿ = Field{Center, Center, Nothing}(grid),
               Jˢⁿ = Field{Center, Center, Nothing}(grid))

    native = atmosphere.grid
    arch = architecture(grid)

    regridder = on_architecture(arch, ConservativeRegridding.Regridder(grid, native))
    raw     = (; speed = CenterField(native),
             direction = CenterField(native), 
                     T = CenterField(native),
                    rh = CenterField(native), 
                     p = CenterField(native),
                precip = CenterField(native), 
              snowfall = CenterField(native))

    derived = (; u = CenterField(native), 
                 v = CenterField(native), 
                 q = CenterField(native),
              rain = CenterField(native))

    regridding = CERRAAtmosphereRegridder(regridder, raw, derived)
    correction = materialize_correction(correction, grid, atmosphere)
    return ComponentExchanger(state, regridding, correction)
end

# The conservative regridder is built eagerly above; nothing to precompute here.
initialize!(exchanger::ComponentExchanger, grid, atmosphere::CurvilinearAtmosphere) = nothing

function interpolate_state!(exchanger::ComponentExchanger, grid, atmosphere::CurvilinearAtmosphere, coupled_model)
    ex    = exchanger.regridder
    state = exchanger.state
    reg, raw, derived = ex.regridder, ex.raw, ex.derived
    t = Time(coupled_model.clock.time)

    # Time-interpolate the raw CERRA fields onto the native grid (slot convention set in
    # `CERRAPrescribedAtmosphere`: u = wind speed, v = wind direction, q = relative humidity,
    # freshwater_flux.rain = total precipitation).
    parent(raw.speed)     .= parent(atmosphere.velocities.u[t])
    parent(raw.direction) .= parent(atmosphere.velocities.v[t])
    parent(raw.T)         .= parent(atmosphere.tracers.T[t])
    parent(raw.rh)        .= parent(atmosphere.tracers.q[t])
    parent(raw.p)         .= parent(atmosphere.pressure[t])
    parent(raw.precip)    .= parent(atmosphere.freshwater_flux.rain[t])   # total precipitation
    parent(raw.snowfall)  .= parent(atmosphere.freshwater_flux.snow[t])   # snowfall water equivalent

    # Derive u, v, q, and liquid rain (= total precipitation − snowfall) on the native grid (vectorized
    # over the field interiors → a GPU kernel on the GPU). The derivations are nonlinear, so they must
    # precede the (linear) conservative regrid; snowfall is regridded directly.
    interior(derived.u)    .= wind_u.(interior(raw.speed), interior(raw.direction))
    interior(derived.v)    .= wind_v.(interior(raw.speed), interior(raw.direction))
    interior(derived.q)    .= specific_humidity.(interior(raw.rh), interior(raw.T), interior(raw.p))
    interior(derived.rain) .= max.(interior(raw.precip) .- interior(raw.snowfall), 0)
    foreach(fill_halo_regions!, (derived.u, derived.v, derived.q, derived.rain))

    # Conservatively regrid each native field into the exchange-grid state.
    ConservativeRegridding.regrid!(state.u,   reg, derived.u)
    ConservativeRegridding.regrid!(state.v,   reg, derived.v)
    ConservativeRegridding.regrid!(state.T,   reg, raw.T)
    ConservativeRegridding.regrid!(state.q,   reg, derived.q)
    ConservativeRegridding.regrid!(state.p,   reg, raw.p)
    ConservativeRegridding.regrid!(state.Jʳⁿ, reg, derived.rain)
    ConservativeRegridding.regrid!(state.Jˢⁿ, reg, raw.snowfall)

    # The exchange grid is a lat–lon grid in the geographic frame, matching CERRA's eastward/northward
    # velocities, so no intrinsic-vector rotation is needed here.
    potential = forcing_barotropic_potential(coupled_model.ocean)
    if !isnothing(potential)
        ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
        parent(potential) .= parent(state.p.data) ./ ρᵒᶜ
    end

    return nothing
end

#####
##### The same treatment for the downwelling-radiation component on the native Lambert grid
#####

const CurvilinearRadiation = PrescribedRadiation{<:OrthogonalSphericalShellGrid}

function ComponentExchanger(radiation::CurvilinearRadiation, grid)
    state = (; ℐꜜˢʷ = Field{Center, Center, Nothing}(grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(grid))

    native_grid = radiation.grid
    regridder = on_architecture(architecture(grid), ConservativeRegridding.Regridder(grid, native_grid))
    regridding = ConservativeAtmosphereRegridder(regridder, CenterField(native_grid))
    return ComponentExchanger(state, regridding)
end

initialize!(exchanger::ComponentExchanger, grid, radiation::CurvilinearRadiation) = nothing

function interpolate_state!(exchanger::ComponentExchanger, grid, radiation::CurvilinearRadiation, coupled_model)
    regridding = exchanger.regridder
    state = exchanger.state
    time = coupled_model.clock.time
    regrid_atmosphere_field!(state.ℐꜜˢʷ, regridding, radiation.downwelling_shortwave, time)
    regrid_atmosphere_field!(state.ℐꜜˡʷ, regridding, radiation.downwelling_longwave,  time)
    return nothing
end

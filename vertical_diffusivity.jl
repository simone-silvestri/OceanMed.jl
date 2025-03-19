using Oceananigans
using Oceananigans.Architectures: architecture
using Oceananigans.BuoyancyModels: ∂z_b
using Oceananigans.Operators
using Oceananigans.BoundaryConditions
using Oceananigans.Grids: inactive_node
using Oceananigans.Grids: total_size
using Oceananigans.Utils: KernelParameters
using Oceananigans.Operators: ℑzᵃᵃᶜ, ℑxyᶠᶠᵃ, ℑxyᶜᶜᵃ

using Adapt

using KernelAbstractions: @index, @kernel
using KernelAbstractions.Extras.LoopInfo: @unroll

using Oceananigans.TurbulenceClosures:
        AbstractScalarDiffusivity,
        ExplicitTimeDiscretization

import Oceananigans.TurbulenceClosures:
        compute_diffusivities!,
        DiffusivityFields,
        viscosity, 
        diffusivity,
        getclosure

using Oceananigans.Utils: launch!
using Oceananigans.Coriolis: fᶠᶠᵃ
using Oceananigans.Operators
using Oceananigans.BuoyancyModels: ∂x_b, ∂y_b, ∂z_b 

using Oceananigans.TurbulenceClosures
using Oceananigans.TurbulenceClosures: HorizontalFormulation, VerticalFormulation
using Oceananigans.TurbulenceClosures: AbstractScalarBiharmonicDiffusivity
using Oceananigans.Operators
using Oceananigans.Operators: Δxᶜᶜᶜ, Δyᶜᶜᶜ, ℑxyᶜᶜᵃ, ζ₃ᶠᶠᶜ, div_xyᶜᶜᶜ
using Oceananigans.Operators:  ℑyzᵃᶜᶠ, ℑxzᶜᵃᶠ, Δxᶜᶜᶜ, Δyᶜᶜᶜ

struct CalibratedRiBasedVerticalDiffusivity{TD, FT} <: AbstractScalarDiffusivity{TD, VerticalFormulation, 2}
    ν₀  :: FT
    νˢʰ :: FT
    νᶜⁿ :: FT
    Cᵉⁿ :: FT
    Prc :: FT
    Prs :: FT
    Riᶜ :: FT
    δRi :: FT
    Q₀  :: FT
    δQ  :: FT
end

function CalibratedRiBasedVerticalDiffusivity{TD}(ν₀  :: FT, 
                                                  νˢʰ :: FT,
                                                  νᶜⁿ :: FT,
                                                  Cᵉⁿ :: FT,
                                                  Prc :: FT,
                                                  Prs :: FT,
                                                  Riᶜ :: FT,
				                                  δRi :: FT,
                                                  Q₀  :: FT,
	         		                              δQ  :: FT) where {TD, FT}
                                       
    return CalibratedRiBasedVerticalDiffusivity{TD, FT}(ν₀, νˢʰ, νᶜⁿ, Cᵉⁿ, Prc, Prs, Riᶜ, δRi, Q₀, δQ)
end

function CalibratedRiBasedVerticalDiffusivity(time_discretization = VerticallyImplicitTimeDiscretization(),
                                              FT  = Float64;
				                              ν₀  = 1e-5, 
                                              νˢʰ = 0.07738088203341657,
                                              νᶜⁿ = 0.533741914196933,
                                              Cᵉⁿ = 0.5196272898085122,
                                              Prc = 0.01632117727992826,
                                              Prs = 1.8499159986192901,
                                              Riᶜ = 0.4923581673007292,
				                              δRi = 0.00012455519496760374,
                                              Q₀  = 0.048232078296680234,
	         		                          δQ  = 0.01884938627051353) 

    TD = typeof(time_discretization)

    return CalibratedRiBasedVerticalDiffusivity{TD}(convert(FT, ν₀),
                                                    convert(FT, νˢʰ),
                                                    convert(FT, νᶜⁿ),
                                                    convert(FT, Cᵉⁿ),
                                                    convert(FT, Prc),
                                                    convert(FT, Prs),
                                                    convert(FT, Riᶜ),
					                                convert(FT, δRi),
					                                convert(FT, Q₀),
					                                convert(FT, δQ))
end

CalibratedRiBasedVerticalDiffusivity(FT::DataType; kw...) =
    CalibratedRiBasedVerticalDiffusivity(VerticallyImplicitTimeDiscretization(), FT; kw...)

Adapt.adapt_structure(to, clo::CalibratedRiBasedVerticalDiffusivity{TD, FT}) where {TD, FT} = 
    CalibratedRiBasedVerticalDiffusivity{TD, FT}(clo.ν₀, clo.νˢʰ, clo.νᶜⁿ, clo.Cᵉⁿ, clo.Prc, clo.Prs, clo.Riᶜ, clo.δRi, clo.Q₀, clo.δQ)   	
                                       
#####                                    
##### Diffusivity field utilities        
#####                                    
                                         
const RBVD = CalibratedRiBasedVerticalDiffusivity   
const RBVDArray = AbstractArray{<:RBVD}
const FlavorOfRBVD = Union{RBVD, RBVDArray}
const c = Center()
const f = Face()

@inline viscosity_location(::FlavorOfRBVD)   = (c, c, f)
@inline diffusivity_location(::FlavorOfRBVD) = (c, c, f)

@inline viscosity(::FlavorOfRBVD, diffusivities) = diffusivities.κᵘ
@inline diffusivity(::FlavorOfRBVD, diffusivities, id) = diffusivities.κᶜ

with_tracers(tracers, closure::FlavorOfRBVD) = closure

# Note: computing diffusivities at cell centers for now.
function DiffusivityFields(grid, tracer_names, bcs, ::FlavorOfRBVD)
    κc = Field((Center, Center, Face), grid)
    κu = Field((Center, Center, Face), grid)
    N² = Field((Center, Center, Face), grid)
    Ri = Field((Center, Center, Face), grid)
    return (; κc, κu, Ri, N²)
end

function compute_diffusivities!(diffusivities, closure::FlavorOfRBVD, model; parameters = :xyz)
    arch = model.architecture
    grid = model.grid
    clock = model.clock
    tracers = model.tracers
    buoyancy = model.buoyancy
    velocities = model.velocities
    top_tracer_bcs = NamedTuple(c => tracers[c].boundary_conditions.top for c in propertynames(tracers))

    launch!(arch, grid, parameters, _compute_Ri_and_stratification!, diffusivities, grid, closure, velocities, tracers, buoyancy)

    # Use `only_local_halos` to ensure that no communication occurs during
    # this call to fill_halo_regions!
    fill_halo_regions!(diffusivities.Ri; only_local_halos=true)

    launch!(arch, grid, parameters,
            _compute_Ri_based_diffusivities!,
            diffusivities,
            grid,
            closure,
            velocities,
            tracers,
            buoyancy,
            top_tracer_bcs,
            clock)

    return nothing
end

@inline ϕ²(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ², ∂zᶠᶜᶠ, velocities.u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ², ∂zᶜᶠᶠ, velocities.v)
    return ∂z_u² + ∂z_v²
end

@inline function Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
    S² = shear_squaredᶜᶜᶠ(i, j, k, grid, velocities)
    Ri = N² / S²

    # Clip N² and avoid NaN
    return ifelse(N² == 0, zero(grid), Ri)
end

const c = Center()
const f = Face()

@kernel function _compute_Ri_and_stratification!(diffusivities, grid, ::FlavorOfRBVD, velocities, tracers, buoyancy)
    i, j, k = @index(Global, NTuple)
    N² = ∂z_b(i, j, k, grid, buoyancy, tracers)
    @inbounds diffusivities.N²[i, j, k] = N²
    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, N²)
end

@kernel function _compute_Ri_based_diffusivities!(diffusivities, grid, closure::FlavorOfRBVD,
                                                velocities, tracers, buoyancy, tracer_bcs, clock)
    i, j, k = @index(Global, NTuple)
    compute_Ri_based_diffusivities!(i, j, k, diffusivities, grid, closure,
                                    velocities, tracers, buoyancy, tracer_bcs, clock)
end

@inline function compute_Ri_based_diffusivities!(i, j, k, diffusivities, grid, closure,
                                                 velocities, tracers, buoyancy, tracer_bcs, clock)

    # Ensure this works with "ensembles" of closures, in addition to ordinary single closures
    closure_ij = getclosure(i, j, closure)

    ν₀  = closure_ij.ν₀  
    νˢʰ = closure_ij.νˢʰ
    νᶜⁿ = closure_ij.νᶜⁿ
    Cᵉⁿ = closure_ij.Cᵉⁿ
    Riᶜ = closure_ij.Riᶜ
    δRi = closure_ij.δRi
    Q₀  = closure_ij.Q₀ 
    δQ  = closure_ij.δQ
    Prc = closure_ij.Prc
    Prs = closure_ij.Prs

    κ₀  = ν₀  / Prs
    κˢʰ = νˢʰ / Prs
    κᶜⁿ = νᶜⁿ / Prc

    Qᵇ = top_buoyancy_flux(i, j, grid, buoyancy, tracer_bcs, clock, merge(velocities, tracers))

    # (Potentially) apply a horizontal filter to the Richardson number
    Ri       = ℑxyᶜᶜᵃ(i, j, k,   grid, ℑxyᶠᶠᵃ, diffusivities.Ri)
    Ri_above = ℑxyᶜᶜᵃ(i, j, k+1, grid, ℑxyᶠᶠᵃ, diffusivities.Ri)
    N²       = ℑxyᶜᶜᵃ(i, j, k,   grid, ℑxyᶠᶠᵃ, diffusivities.N²)
    
    # Conditions
    convecting = Ri < 0 # applies regardless of Qᵇ
    entraining = (Ri > 0) & (Ri_above < 0) & (Qᵇ > 0)

    # Convective adjustment diffusivity
    ν_local = ifelse(convecting, (νˢʰ - νᶜⁿ) * tanh(Ri / δRi) + νˢʰ, clamp((ν₀ - νˢʰ) * Ri / Riᶜ + νˢʰ, ν₀, νˢʰ))
    κ_local = ifelse(convecting, (κˢʰ - κᶜⁿ) * tanh(Ri / δRi) + κˢʰ, clamp((κ₀ - κˢʰ) * Ri / Riᶜ + κˢʰ, κ₀, κˢʰ))

    # Entrainment diffusivity
    ϵ = eps(eltype(grid))
    x = Qᵇ / (N² +  ϵ)
    ν_nonlocal = ifelse(entraining,  Cᵉⁿ * νᶜⁿ * (tanh((x - Q₀) / δQ) + 1) / 2, 0)
    κ_nonlocal = ifelse(entraining,  ν_nonlocal / Prs, 0)

    # Update by averaging in time
    @inbounds diffusivities.κᵘ[i, j, k] = ν_local + ν_nonlocal
    @inbounds diffusivities.κᶜ[i, j, k] = κ_local + κ_nonlocal

    return nothing
end
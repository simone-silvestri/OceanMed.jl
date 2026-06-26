module OceanMed

include("vertical_grids.jl")
include("open_boundary_conditions.jl")
include("bathymetry_datasets.jl")
include("river_forcing.jl")
include("cerra_forcing.jl")

using .VerticalGrids
using .OpenBoundaryConditions
using .BathymetryDatasets
using .RiverForcing
using .CERRAForcing

export copernicus_z_faces
export atlantic_boundary_conditions, atlantic_sponge_forcings, AtlanticSpongeMask
export MEDSEABathymetry, EMODnetBathymetry
export CERRAReanalysis, cerra_native_grid, CERRAPrescribedAtmosphere, CERRAPrescribedRadiation
export MediterraneanPrescribedLand

end # module OceanMed

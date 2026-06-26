module OceanMed

include("vertical_grids.jl")
include("open_boundary_conditions.jl")
include("bathymetry_datasets.jl")
include("river_forcing.jl")
include("cerra_forcing.jl")

using .VerticalGrids
using .OpenBoundaryConditions
using .BathymetryDatasets
using .CERRAForcing

export copernicus_z_faces
export gibraltar_boundary_conditions, gibraltar_sponge_forcings, WesternSpongeMask
export MEDSEABathymetry, EMODnetBathymetry
export CERRAReanalysis, cerra_native_grid, CERRAPrescribedAtmosphere, CERRAPrescribedRadiation

end # module OceanMed

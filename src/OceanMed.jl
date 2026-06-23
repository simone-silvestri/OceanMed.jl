module OceanMed

include("vertical_grids.jl")
include("open_boundary_conditions.jl")
include("mediterranean_bathymetry.jl")

using .VerticalGrids
using .OpenBoundaryConditions
using .MediterraneanBathymetry

export copernicus_z_faces
export gibraltar_boundary_conditions, gibraltar_sponge_forcings, WesternSpongeMask
export MEDSEABathymetry

end # module OceanMed

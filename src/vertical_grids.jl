module VerticalGrids

export copernicus_z_faces

using NCDatasets

"""
    copernicus_z_faces(; filepath = "data/zc_copernicus.nc", refinement = 2)

Build the vector of vertical face positions (in metres, negative downward) of the Copernicus
Mediterranean vertical grid, refined by an integer `refinement` factor.

The file `zc_copernicus.nc` stores the depths of the layer centres. The corresponding faces are
first reconstructed from those centres, then each layer is split into `refinement` equal
sublayers. The default `refinement = 2` reproduces the 280-level grid used by the Mediterranean
simulation (140 native Copernicus layers, each halved).
"""
function copernicus_z_faces(; filepath = "data/zc_copernicus.nc", refinement = 2)
    centers = Dataset(filepath) do dataset
        reverse(dataset["depth"][:])
    end

    Nc = length(centers)
    coarse_faces = zeros(Nc + 1)
    for k in Nc:-1:1
        coarse_faces[k] = coarse_faces[k+1] - (centers[k] + coarse_faces[k+1]) * 2
    end

    faces = zeros(refinement * Nc + 1)
    for k in 1:Nc, r in 0:refinement-1
        faces[refinement * (k - 1) + r + 1] = coarse_faces[k] + (coarse_faces[k+1] - coarse_faces[k]) * r / refinement
    end
    faces[end] = coarse_faces[end]

    return faces
end

end # module VerticalGrids

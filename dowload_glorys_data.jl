using ClimaOcean
using Oceananigans

ds = Dataset("data/mesh_mask2D_NA.nc")
xc = ds["XC"][:]
yc = ds["YC"][:]

x_faces = [xc..., xc[end] + (xc[end] - xc[end-1])]
y_faces = [yc..., yc[end] + (yc[end] - yc[end-1])]

# Load the forcing data
start_date = Date(2017, 1, 1)

bbox = ClimaOcean.DataWrangling.BoundingBox(
            longitude = (x_faces[1]-2, x_faces[end]+2), 
            latitude  = (y_faces[1]-2, y_faces[end]+2))

dir = "./data"

# Load the forcing data
start_date = Date(2017, 1, 1)

bbox = ClimaOcean.DataWrangling.BoundingBox(
            longitude = (x_faces[1]-2, x_faces[end]+2), 
            latitude  = (y_faces[1]-2, y_faces[end]+2))

dir = "./data"

dataset = GLORYSDaily()
u_meta = Metadata(:u_velocity;  dataset, dir, bounding_box=bbox, start_date)
v_meta = Metadata(:v_velocity;  dataset, dir, bounding_box=bbox, start_date)
T_meta = Metadata(:temperature; dataset, dir, bounding_box=bbox, start_date)
S_meta = Metadata(:salinity;    dataset, dir, bounding_box=bbox, start_date)

# Actually download the data, it will appear in the "./data" folder
ClimaOcean.DataWrangling.download_dataset(u_meta)
ClimaOcean.DataWrangling.download_dataset(v_meta)
ClimaOcean.DataWrangling.download_dataset(T_meta)
ClimaOcean.DataWrangling.download_dataset(S_meta)

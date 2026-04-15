# Download GLORYS data for the Mediterranean simulation OBCs
using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, download_dataset
using Oceananigans
using Dates

const λ₁, λ₂ = (-8.6, 42)
const φ₁, φ₂ = (30, 48)

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

# Download the data (requires Copernicus Marine Service credentials)
download_dataset(u_meta)
download_dataset(v_meta)
download_dataset(T_meta)
download_dataset(S_meta)

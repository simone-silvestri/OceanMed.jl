using ClimaOcean
using ClimaOcean.ECCO
using Dates

dataset = ECCO4Monthly()
dates = all_dates(dataset)

temperature = Metadata(:temperature; dates, dataset, dir="./data")
salinity = Metadata(:salinity; dates, dataset, dir="./data")

download_dataset(temperature)
download_dataset(salinity)

JRA55PrescribedAtmosphere(dir="./data")

grid = LatitudeLongitudeGrid(size=(10, 10, 10), latitude=(-10, 10), longitude=(-10, 10), z=(-1, 0))
regrid_bathymetry(grid; dir="./data")

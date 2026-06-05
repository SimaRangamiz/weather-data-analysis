import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["total_precipitation"],
    "year": [
        "1989", "1990", "1991", 
        "1992", "1993", "1994", 
        "1995", "1996", "1997", 
        "1998", "1999", "2000"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [37.2, 45.2, 32.5, 50.7]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

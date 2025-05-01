from osgeo import gdal, ogr

def rasterize_mask(vector_path, ref_raster_path, output_path):
    # Open reference raster (Delft)
    ref_ds = gdal.Open(ref_raster_path)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize

    # Create output raster
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, x_size, y_size, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(1)
    out_band.Fill(0)

    # Open vector and rasterize
    vector_ds = ogr.Open(vector_path)
    layer = vector_ds.GetLayer()
    gdal.RasterizeLayer(out_ds, [1], layer, burn_values=[1])

    out_ds.FlushCache()
    print(f"Done: {output_path}")

# Example usage
rasterize_mask(
    vector_path=r"D:\Super_Resolution\data\buildings\buildings_delft.shp",
    ref_raster_path=r"D:\Super_Resolution\data\rasters\Delft_lr.tif",
    output_path=r"D:\Super_Resolution\data\rasters\building_mask_delft.tif"
)

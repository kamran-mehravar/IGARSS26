import rasterio

CUBE_TIF  = r"D:\USA_paper\dataset\emit cuprite\emit cuprite\resize-continuum.tif"
LABEL_TIF = r"D:\USA_paper\dataset\Emit Py folder\klabels10_georef.tif"

def show(path, name):
    with rasterio.open(path) as src:
        print(f"\n--- {name} ---")
        print("shape:", (src.height, src.width))
        print("crs:", src.crs)
        print("transform:", src.transform)
        print("bounds:", src.bounds)

show(CUBE_TIF,  "CUBE")
show(LABEL_TIF, "LABEL")

import rasterio
import numpy as np
import matplotlib.pyplot as plt


image_path = r"D:\USA_paper\dataset\emit cuprite\emit cuprite\resize tiff.tif" # فایل تصویر اصلی
diff_map_path = r"D:\USA_paper\result\map\diff_1d_3d.tif" # فایل مپ اختلاف که ساختید


with rasterio.Env(GEOREF_SOURCES="INTERNAL"):
    with rasterio.open(image_path, nodata=0) as src:
    
        fc = src.read([50, 30, 10]).astype(np.float32)


for i in range(3):
    min_val = fc[i].min()
    max_val = fc[i].max()
    if max_val - min_val > 0:
        fc[i] = (fc[i] - min_val) / (max_val - min_val)
    else:
        fc[i] = 0


fc = fc.transpose(1, 2, 0)


with rasterio.open(diff_map_path) as src:
    diff_data = src.read(1) 


rows, cols = diff_data.shape

overlay = np.zeros((rows, cols, 4), dtype=np.float32)


diff_indices = (diff_data == 1)


overlay[diff_indices] = [1.0, 1.0, 0.0, 0.5] 


plt.figure(figsize=(12, 10))


plt.imshow(fc)


plt.imshow(overlay)

plt.axis("off")
plt.title("Overlay of Difference Map on EMIT False Color\n(Yellow = Difference, Alpha = 0.5)")
plt.show()
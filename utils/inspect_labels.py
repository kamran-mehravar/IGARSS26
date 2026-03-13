import numpy as np
import rasterio
from collections import Counter

LABEL = r"D:\USA_paper\dataset\Emit Py folder\klabels10_georef.tif"

with rasterio.open(LABEL) as src:
    lab = src.read(1)

vals, cnts = np.unique(lab, return_counts=True)
print("Unique values:", vals.tolist())
print("Counts:", dict(zip(vals.tolist(), cnts.tolist())))

# حدس‌های رایج برای nodata:
candidates = [255, 254, -1, 65535]
present = [v for v in candidates if v in vals]
print("Common nodata candidates present:", present)

# اگر مقدار nodata در متادیتا ثبت شده باشد:
with rasterio.open(LABEL) as src:
    print("Rasterio nodata:", src.nodata)

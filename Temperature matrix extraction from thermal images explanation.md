**Download DJI Thermal SDK** from: 

https://www.dji.com/ca/downloads/softwares/dji-thermal-sdk





Open PowerShell in the **DJI Thermal SDK (dji\_thermal\_sdk\_v1.8\_20250829)** 



Find this path:

* "path where it was downloaded\\dji\_thermal\_sdk\_v1.8\_20250829\\utility\\bin\\windows\\release\_x64"
* Right click anywhere in the file folder (except for an actual file) -> open in terminal (Power Shell)



**In PowerShell (PS) write (if needed, adjust settings based on the explanations I give below)**



.\\dji\_irp.exe -s "C:\\path\\to\\DJI\_0056\_T.JPG" -a measure -o "C:\\path\\to\\DJI\_0056\_T.raw" --measurefmt 1 --distance 5 --humidity 70 --emissivity 0.95 --ambient 25 --reflection 23\\



* measurefmt 1: Output format for the temperature matrix. Format 1 produces signed 16-bit integers representing tenths of a degree Celsius. A stored value of 248 means 24.8 °C. I would recommend keeping this setting at 1 and adjusting the temperature in the code by dividing the stored values by 10
* distance = distance from camera to measured surface, in meters 
* humidity = relative humidity between camera and target
* ambient refers to ambient temperature (in deg C)
* reflection - reflected apparent Temp (in deg C) - represents infrared radiation from the surroundings reflected by the blade
* emissivity - for white surfaces typically between 0.85 and 0.95; must check the emissivity for your surface type



**Full examples (what I use w/ path included) - I recommend changing the bolted paths and settings here, then copying as pasting in PS**



Thermal Image 1 - .\\dji\_irp.exe -s **"C:\\Users\\mihai.dobre\\Downloads\\dji\_thermal\_sdk\_v1.8\_20250829\\utility\\bin\\windows\\release\_x64\\DJI\_0333\_T.JPG"** -a measure -o **"C:\\Users\\mihai.dobre\\Downloads\\dji\_thermal\_sdk\_v1.8\_20250829\\utility\\bin\\windows\\release\_x64\\DJI\_0333\_T.raw"** --measurefmt 1 --distance 5 --humidity 70 --emissivity 0.95 --ambient 25 --reflection 23



Thermal Image 2 - .\\dji\_irp.exe -s **"C:\\Users\\mihai.dobre\\Downloads\\dji\_thermal\_sdk\_v1.8\_20250829\\utility\\bin\\windows\\release\_x64\\DJI\_0335\_T.JPG"** -a measure -o **"C:\\Users\\mihai.dobre\\Downloads\\dji\_thermal\_sdk\_v1.8\_20250829\\utility\\bin\\windows\\release\_x64\\DJI\_0335\_T.raw"** --measurefmt 1 --distance 5 --humidity 70 --emissivity 0.95 --ambient 25 --reflection 23





**Convert to NumPy matrix (check if temperature matrix makes sense)**

**Note: The output image might not look the exact same because the program will likely pick a different temperature gradient than the one used by the drone**







import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path



downloads = Path(r"C:\\Users\\mihai.dobre\\Downloads\\dji\_thermal\_sdk\_v1.8\_20250829\\utility\\bin\\windows\\release\_x64")



files = \[

&#x20;   downloads / "DJI\_0097\_T.raw",

&#x20;   downloads / "DJI\_0099\_T.raw"

]



WIDTH = 640

HEIGHT = 512



for raw\_file in files:

&#x20;   temp\_raw = np.fromfile(raw\_file, dtype=np.int16)



&#x20;   print(raw\_file.name)

&#x20;   print("Values:", temp\_raw.size)

&#x20;   print("Expected:", WIDTH \* HEIGHT)



&#x20;   if temp\_raw.size != WIDTH \* HEIGHT:

&#x20;       print("Unexpected size!")

&#x20;       continue



&#x20;   # DJI int16 output is usually temperature × 10 - look at the output pictures and decide if the scale factor is correct, otherwise adjust it

&#x20;   temp\_c = temp\_raw.reshape((HEIGHT, WIDTH)).astype(np.float32) / 10.0



&#x20;   np.save(raw\_file.with\_suffix(".npy"), temp\_c)



&#x20;   plt.figure(figsize=(10, 7))

&#x20;   plt.imshow(temp\_c, cmap="inferno")

&#x20;   plt.colorbar(label="Temperature (°C)")

&#x20;   plt.title(raw\_file.stem)

&#x20;   plt.axis("off")

&#x20;   plt.tight\_layout()

&#x20;   plt.savefig(raw\_file.with\_suffix(".png"), dpi=300)

&#x20;   plt.show()



&#x20;   print("Min °C:", np.nanmin(temp\_c))

&#x20;   print("Max °C:", np.nanmax(temp\_c))

&#x20;   print("Saved:", raw\_file.with\_suffix(".npy"))

&#x20;   print("Saved:", raw\_file.with\_suffix(".png"))


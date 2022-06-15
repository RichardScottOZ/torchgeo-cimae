"""."""
# %%
import os

from torchgeo.datasets import NAIP, create_bounding_box

# %%
area_of_interest = create_bounding_box(
    -120.3, -119.4, 36.4, 37.1, "2016-01-01", "2017-01-01"
)
date_range = "2016-01-01/2017-01-01"
# %%
NAIP(
    os.path.join(os.environ["SCRATCH"], "NAIP_big"),
    area_of_interest=area_of_interest,
    date_range=date_range,
    download=True,
)

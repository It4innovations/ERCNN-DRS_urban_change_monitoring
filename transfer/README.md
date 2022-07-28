# Transfer Learning with ERCNN-DRS
TBD

# Training/Validation Data
Thanks to the data providers, we can make available the [`training/validation dataset`](https://drive.google.com/drive/folders/1CLTna5fNLTEEWwELK6hXoN5C42yaXvQf?usp=sharing) on Google Drive.

**Note:** The tiles don't contain the ground truth but synthetic labels which are not used for transfer learning! The ground truth needs to be loaded separately (see folder [`numpy_ground_truth`](./numpy_ground_truth/)).

**ATTENTION, these files are large: 124-140 GB**
   
**Sentinel 1 & 2, AoI Liège:**
- [baseline](https://drive.google.com/file/d/1h5aZCnXoAgZU8ZqiZVwB8Q99iR0LwWLw/view?usp=sharing) [124 GB]
- [extended](https://drive.google.com/file/d/1JzSpCUmPpAKYP5P2RS3ZYsVN8sgQY107/view?usp=sharing) [140 GB]

# Trained Models
We provide three [`models`](./models/):
  - [`epoch_0.hdf5`](./models/epoch_0.hdf5): Original, pre-trained model.
  - [`baseline.hdf5`](./models/baseline.hdf5): Transferred model with baseline dataset.
  - [`extended.hdf5`](./models/extended.hdf5): Transferred model with extended dataset.

# Paper and Citation
TBD

# Contact
Should you have any feedback or questions, please contact the main author: Georg Zitzlsberger (georg.zitzlsberger(a)vsb.cz).

# Acknowledgments
This research was funded by the IT4Innovations infrastructure which is supported from the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140) via Open Access Grant Competition (OPEN-21-31). The authors would like to thank the data providers (Sentinel Hub and Google) for making the used remote sensing data freely available:
- Contains modified Copernicus Sentinel data 2017-2021 processed by Sentinel Hub (Sentinel 1 & 2).


# License
This project is made available under the GNU General Public License, version 3 (GPLv3).

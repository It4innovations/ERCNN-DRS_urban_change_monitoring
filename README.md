# ERCNN-DRS Urban Change Monitoring
This project hosts the Ensemble of Recurrent Convolutional Neural Networks for Deep Remote Sensing (ERCNN-DRS) used for urban change monitoring with ERS-1/2 & Landsat 5 TM, and Sentinel 1 & 2 remote sensing mission pairs. It was developed for demonstration purposes in the ESA Blockchain ENabled DEep Learning for Space Data (BLENDED)<sup>1</sup> project.
Two neural network models were trained for the two eras (ERS-1/2 & Landsat 5 TM: 1991-2011, and Sentinel 1 & 2: 2017-2021).

## Features
- Trained with SAR and optical multispectral observation time series of hundreds up to thousands of observations (deep-temporal)
- Demonstrates usage for two mission pairs: ERS-1/2 & Landsat 5 TM (1991-2011), and 6m for Sentinel 1 & 2 (2017-now)
- Predicts changes which happened in one time window (1y for ERS-1/2 & Landsat 5 TM, and 6m for Sentinel 1 & 2)
- The long mission times allow monitoring of urban changes over larger time frames

## Usage
### Pre-Requisites
For either era, SAR (separated by ascending and descending orbit directions) and multispectral optical observations are needed as EOPatches, a format introduced by [eo-learn](https://github.com/sentinel-hub/eo-learn).

### Pre-Processing
Before training, observations from EOPatches need to be processed in two steps:
<!--![pre-processing steps](./collateral/pre-processing.png)-->

1. Temporally stacking, assembling and tiling (creates temporary TFRecord files):
    - ERS-1/2 & Landsat 5 TM: [`1_tstack_assemble_tile.py`](./ERS12_LS5/preproc/)
    - Sentinel 1 & 2: [`1_tstack_assemble_tile.py`](./Sentinel1_2/preproc/)
2. Windowing and labeling (output: TFRecord files):
    - ERS-1/2 & Landsat 5 TM: [`2_generate_windows_slabels.py`](./ERS12_LS5/preproc/)
    - Sentinel 1 & 2: [`2_generate_windows_slabels.py`](./Sentinel1_2/preproc/)

### Model Architecture
<!--![model architecture](./collateral/model_architecture.png)-->

### Training
Training is executed on the windowed and labeled TFRecord files:
  - ERS-1/2 & Landsat 5 TM: [`train.py`](./ERS12_LS5/train/)
  - Sentinel 1 & 2: [`train.py`](./Sentinel1_2/train/)

### Inference
We provide pre-trained networks which can be used right away:
  - ERS-1/2 & Landsat 5 TM: [`best_weights_ercnn_drs.hdf5`](./ERS12_LS5/train/snapshots/)
  - Sentinel 1 & 2: [`best_weights_ercnn_drs.hdf5`](./Sentinel1_2/train/snapshots/)

## Examples
<!--### ERS-1/2 & Landsat 5 TM
ERS-1/2 & Landsat 5 TM example of Liège. Top row are Landsat 5 TM true color observations (left, right) with change prediction (middle). Bottom rows are corresponding very-high resolution imagery from Google Earth(tm), (c) 2021 Maxar Technologies with predictions superimposed in red.

![Sentinel 1 & 2 urban changes](./collateral/ers12ls5_example.png)

 Series of predictions from above example.

![Sentinel 1 & 2 urban changes time series](./collateral/ers12ls5_example_series.png)

### Sentinel 1 & 2
Sentinel 1 & 2 example of Liège. Top row are Sentinel 2 true color observations (left, right) with change prediction (middle). Bottom rows are corresponding very-high resolution imagery from Google Earth(tm), (c) 2021 Maxar Technologies with predictions superimposed in red.

![Sentinel 1 & 2 urban changes](./collateral/s12_example.png)

 Series of predictions from above example.

![Sentinel 1 & 2 urban changes time series](./collateral/s12_example_series.png)-->

# Paper and Citation
TBD

# Contact
Should you have any feedback or questions, please contact the main author: Georg Zitzlsberger (georg.zitzlsberger(a)vsb.cz).

# Acknowledgement
This research was funded by ESA via the Blockchain ENabled DEep Learning for Space Data (BLENDED) project (SpaceApps Subcontract No. 4000129481/19/I-IT4I), and by the Ministry of Education, Youth and Sports from the National Programme of Sustainability (NPS II) project "IT4Innovations excellence in science - LQ1602" and by the IT4Innovations infrastructure whichsupported from the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90140) via Open Access Grant Competition (OPEN-21-31).

The authors would like to thank ESA for funding of the study as part of the BLENDED project<sup>1</sup> and IT4Innovations for funding the compute resources via the Open Access Grant Competition (OPEN-21-31). Furthermore, the authors would like to thank the data providers (USGS, ESA, Sentinel Hub and Google) for making remote sensing data freely available.  The authors would finally like to thank the BLENDED project partners for supporting our work as study case of the developed platform.

<sup>1</sup> [Valentin, B.; Gale, L.; Boulahya, H.; Charalampopoulou, B.; Christos K., C.; Poursanidis, D.; Chrysoulakis, N.; Svato&#x0148;, V.; Zitzlsberger, G.; Podhoranyi, M.; Kol&#x00E1;&#x0159;, D.; Vesel&#x00FD;, V.; Lichtner, O.; Koutensk&#x00FD;, M.; Reg&#x00E9;ciov&#x00E1;, D.; M&#x00FA;&#x010D;ka, M. BLENDED - USING BLOCKCHAIN AND DEEP LEARNING FOR SPACE DATA PROCESSING. Proceedings of the 2021 conference on Big Data from Space; Soille, P.; Loekken, S.; Albani, S., Eds. Publications Office of the European Union, 2021, JRC125131, pp. 97-100.  doi:10.2760/125905.](https://op.europa.eu/en/publication-detail/-/publication/ac7c57e5-b787-11eb-8aca-01aa75ed71a1)


# License
This project is made available under the GNU General Public License, version 3 (GPLv3).

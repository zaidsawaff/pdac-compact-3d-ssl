# Third-Party Data Notice

This repository contains study-generated software, metadata, protocol locks,
quality-control artifacts, and reproducibility utilities.

**Raw third-party medical imaging data are not distributed under the MIT
license of this repository.**

## PANORAMA

The study uses CT imaging and segmentation annotations originating from
the public PANORAMA resources.

The reproducibility utilities retrieve or access these materials from their
original public distribution locations rather than redistributing the raw
medical imaging data as repository-owned content.

The exact PANORAMA label repository state used by the study is frozen at:

```text
commit bf1d6ba3
```

Expected annotation counts are:

- Manual labels: 482
- Automatic labels: 1,756
- Total: 2,238

PANORAMA CT volumes are accessed from the original public Zenodo archives
using the study-generated frozen remote ZIP-member inventory and HTTP
byte-range requests.

PANORAMA materials remain governed by the terms specified by their original
authors and distribution records, including CC BY-NC 4.0 where applicable.

## Medical Segmentation Decathlon (MSD)

The external evaluation workflow includes the public pancreatic CT data used
from the Medical Segmentation Decathlon source.

These data are not redistributed under this repository's MIT license.
Users must obtain and use them according to the original dataset terms.

## NIH pancreatic CT data

The study also uses a public NIH pancreatic CT cohort as an external
negative/stress-test source.

These data are not redistributed under this repository's MIT license.
Users must obtain and use them according to the original distribution terms.

## License scope

The MIT license in this repository applies only to software and other
study-generated material for which the repository authors hold the relevant
rights.

It does not replace, modify, sublicense, or override the licenses, access
conditions, attribution requirements, or other terms associated with
third-party datasets.

Researchers reproducing the study are responsible for reviewing and complying
with the terms of each original data source.

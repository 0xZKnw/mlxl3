# PonyExl3 conversion runtime

Source: https://github.com/beamivalice/PonyExl3
Revision: 8e7fa6b1556f59fc669e25087903b279b9b0346f
License and attribution: LICENSE, NOTICE, and ponyexl3/convert/calibration_data/NOTICE.

This snapshot includes the repository's existing patches:
- scripts/patches/ponyexl3-mlxl3-conversion.patch
The consolidated patch includes the bounded conversion cache and earlier
measurement/LDL/GSS changes. Do not reapply the incremental patches on top.

No personal development-checkout modifications are included. Desktop uses the
existing MLXL3 LFM2, LFM2-MoE and Ling adapters; this snapshot is not a claim
that every architecture supported by inference can be converted.

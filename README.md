Overview
This python package contains a complete workflow and data managment suite to bring raw images of brain sections all the way through network analysis using ABBA/QuPath to align to a reference atlas.

Stages
1. Preprocessing - prepare raw image for ABBA
    - Configure data organization scheme
    - Extract downsampled images apply order, rotate, flip, proofread, and rename images to unified format
    - Extract/convert raw fullsize images to OME-TIFF + apply formatting
3. Alignment
    - create Qupath projects for each animal, align in ABBA, export back to QuPath, then run QuPath scripts to export registrations to open format
7. Nuclei detection
    - Nuclei quantification using StarDist, perform colocalization across image channels, and localize nuclei to atlas coordinates 
9. Data compilation
    - Threshold nuclei based on morphological properties, reduce nuclei detections to counts per region/hemisphere
11. Analysis
    - Graph nuclei species per region across conditions (treatment, sex, etc.)
    - PCA analysis to isolate differences between conditional groups
    - Network analysis and interactive visualization - battery of graph analytics using NetworkX and Bokeh
    - Volumetric renderings of aggregated nuclei counts
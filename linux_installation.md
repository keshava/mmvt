---
layout: installation
title: MMVT Linux Installation
---
1. Install [Blender](https://www.blender.org/download/)
2. If you’re planning to use the preprocessing step:
    1. You should have a python 3.4+ installation. My recommendation is to install [Anaconda](https://www.continuum.io/downloads) with python 3.5. After the installation, 
    Make sure pip is anaconda pip (which pip), and install the following packages:
       * ``pip install mne –upgrade``
       * ``pip install pip install nibabel``
    2. Install the dev version of freesurfer from [here](ftp://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/dev).
       If you have access to the Martinos center cluster, you should just source the dev version:
       source /usr/local/freesurfer/nmr-dev-env
3. Clone the mmvt (first install git if you don’t have it):
   git clone [https://github.com/pelednoam/mmvt.git](https://github.com/pelednoam/mmvt.git)
4. Open Blender, File->user preferences (Ctrl Alt U)
   1. Open the Add-ons tab
   2. Install from file…
   3. Select the MMVT_loader.py:
      your_code_folder\mmvt\src\mmvt_addon\mmvt_loader.py
   4. Select the check-box
   5. In the path to the mmvt addon folder write:
      “your_code_folder\mmvt\src\mmvt_addon\”
   6. In path to python write: “...anaconda3_folder\python”
   7. The path to freeview should be “freeview”
   8. If you are using freesurfer dev version, check the two checkboxes. 
      If not, leave them unchecked.
5. Create a script to run blender. An example can be found [here](https://github.com/pelednoam/mmvt/blob/master/misc/launch_blender). 
Run this launcher from the terminal (you can add a shortcut to the desktop / panel). 
This way you’ll be able to see all the warnings / error messages.


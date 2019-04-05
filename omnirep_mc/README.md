# The OMNIREP algorithm: Coevolving encodings and representations

Papers are available through [Moshe Sipper's website](http://www.moshesipper.com/).

Code accompanying the paper, M. Sipper and J. H. Moore, "OMNIREP: Originating meaning by coevolving encodings and representations", *Memetic Computing*, 2019.

* `animated-gifs`: Animated images of art evolved by OMNIREP.   
* `original_target_pics`: Original target images for evolutionary art.

How to run OMNIREP:
1. In `evolve.py` select **one** model import, comment out all the rest, e.g., `import image_and_polygons as modname`     
2. Set parameters in the respective model python file, e.g., `GENERATIONS = 1000` in `image_and_polygons.py`     
3. Create a folder `results`    
4. If you aren’t running images (evolutionary art), you can now run `evolve.py`    
5. For images:    
  a. Create a folder `images`   
  b. Create a subfolder for a particular run, e.g., `images/mypic-run1`     
  c. Place the original **jpg** image in this folder (use pics from `original_target_pics` or your own)   
  d. In the Python model file you’ve selected (e.g, `image_and_polygons.py`) set:      
            `IMAGE_FOLDER = 'images/mypic-run1/' ` (make sure there’s a "/" in the end)    
            `IMAGE_FILE = mypic` (just name, without suffix ".jpg")     
  e. Set parameters, e.g., if using polygons, `NUM_POLYGONS = 50`     
  f. Run `evolve.py`    

import nibabel as nib
from vessel_seg import CoronaryTree

img = nib.load("segmentation.nii.gz")
labels = {"LAD": 1, "LCx": 2, "posterior descending artery": 5}
tree = CoronaryTree()
unmatched = tree.assign_from_labelmap(img.get_fdata(), labels, dtype="uint8")

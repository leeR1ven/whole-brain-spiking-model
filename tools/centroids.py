"""centroids.py <atlas dir> <out json>: centre (MNI mm) and volume (mm^3) of every region of the AAL3 atlas"""
import sys, json, glob, os, numpy as np
import nibabel as nib
d = sys.argv[1]
nii = [f for f in glob.glob(os.path.join(d, '**', '*.nii*'), recursive=True) if 'AAL3v1.nii' in f and '1mm' not in f][0]
lab = [f for f in glob.glob(os.path.join(d, '**', '*.txt'), recursive=True) if f.endswith('AAL3v1.nii.txt')][0]
img = nib.load(nii); data = np.asarray(img.dataobj); aff = img.affine; vox = abs(np.linalg.det(aff[:3, :3]))
out = {}
for line in open(lab):
    t = line.split()
    if len(t) < 2: continue
    k = int(t[0]); ijk = np.argwhere(data == k)
    if len(ijk) == 0: continue
    c = (aff @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]
    out[t[1]] = dict(center=[round(float(v), 1) for v in c.mean(0)], volume_mm3=round(float(len(ijk) * vox), 0))
json.dump(out, open(sys.argv[2], 'w'), indent=0)
print(len(out), 'regions; file', os.path.basename(nii))

"""Where things are. Coordinates in mm (MNI space, left hemisphere) are the centres of the regions of the AAL3 atlas of
the human brain (Rolls et al. 2020; computed from the atlas file, saved in data/aal3_centroids.json).
Which atlas region stands for which model part is MY assignment. Parts with no region in the atlas are placed by hand
next to their neighbours and are marked 'approx'."""
import json, os, numpy as np
_C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/aal3_centroids.json')))
c = lambda k: np.array(_C[k]['center'], dtype=float)
REGION = dict(   # model part -> atlas region
    A1='Heschl_L', AB='Temporal_Sup_L', PB='Temporal_Mid_L', M1i='Rolandic_Oper_L', PMi='Frontal_Inf_Oper_L', PFi='Frontal_Inf_Tri_L',
    V1='Calcarine_L', TO='Fusiform_L', AT='Temporal_Pole_Mid_L', M1L='Precentral_L', PML='Frontal_Sup_2_L', PFL='Frontal_Mid_2_L',
    EC='ParaHippocampal_L', hippocampus='Hippocampus_L', MGB='Thal_MGN_L', LGN='Thal_LGN_L', VL='Thal_VL_L', MD='Thal_MDm_L',
    striatum='Putamen_L', pallidum='Pallidum_L', SNr='SN_pr_L', SNc='SN_pc_L', VTA='VTA_L', amygdala='Amygdala_L', LC='LC_L',
    cerebellum='Cerebellum_Crus1_L', red_nucleus='Red_N_L')
CENTER = {k: c(v) for k, v in REGION.items()}
APPROX = dict(   # not in the atlas: placed by hand
    RT=CENTER['VL'] + np.array([-5.0, 0.0, 0.0]),                    # reticular sheet: just lateral to the thalamus
    STN=(CENTER['SNr'] + CENTER['VL']) / 2 + np.array([-2.0, 0.0, 0.0]),
    DCN=np.array([-17.0, -58.0, -35.0]),                             # deep cerebellar nuclei: in the white matter of the cerebellum
    ear=np.array([-70.0, -25.0, -10.0]), eye=np.array([-32.0, 65.0, -20.0]), mouth=np.array([-10.0, 70.0, -60.0]))
CENTER.update(APPROX)
DENSITY = 77169.0 / 2.0      # cells per mm^3 used for nuclei whose density was not read (= the cortical value used here)


def blob(center, n, rng, density=DENSITY):
    """n cells spread evenly in a ball around center, at the given density"""
    r = (3 * n / (4 * np.pi * density)) ** (1 / 3.0)
    v = rng.standard_normal((n, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return (np.asarray(center) + v * (r * rng.random(n) ** (1 / 3.0))[:, None]).astype(np.float32)

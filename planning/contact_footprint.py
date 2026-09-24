"""Observed full-width cut opportunity and fixed target-mouth markers.

These heightfield quantities rank candidate cuts; they are not retained payload
or a calibrated prediction of fill. No hidden particle/material state is used.
"""
import numpy as np

VERSION = 'full_width_cut_opportunity_v1'


def target_basis(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or np.linalg.norm(xy) < 1e-9:
        raise ValueError('invalid_target_center')
    radial = xy / np.linalg.norm(xy)
    return radial, np.array([-radial[1], radial[0]])


def cut_opportunity(terrain, contact, *, width=.60, length=.90,
                    below_entry=.35, band_height=.70):
    """Integrate an explicit cut band, penalizing its weakest width third.

    Slice bottom is 35 cm below the planned center entry. The 70 cm vertical
    band bounds this local geometric opportunity, rather than clipping world
    heights. Unknown samples invalidate the candidate instead of becoming dirt.
    """
    if not np.isfinite([width,length,below_entry,band_height]).all() or min(width,length,below_entry,band_height)<=0:
        raise ValueError('invalid_cut_band')
    xy = np.asarray(contact['contact_xy_m'], dtype=float)
    radial, lateral = target_basis(xy)
    bottom = float(contact['contact_surface_z_m']) - below_entry
    h = np.asarray(terrain['height_m'], dtype=float)
    count = np.asarray(terrain['point_count'])
    origin = np.asarray(terrain['origin_xy_m'], dtype=float)
    cell = float(terrain['resolution_m'])
    if (not terrain.get('valid') or h.ndim!=2 or count.shape!=h.shape
            or origin.shape!=(2,) or not np.isfinite(origin).all()
            or not np.isfinite([cell,bottom]).all() or cell<=0):
        raise ValueError('invalid_cut_band_terrain')
    # Equal-area samples rotate with the actual approach heading. At least six
    # width bands keep a thin intact center ridge from hiding two empty sides.
    nw=max(6,int(np.ceil(width/(cell/2))))
    nd=max(2,int(np.ceil(length/(cell/2))))
    transverse=(np.arange(nw)+.5)*width/nw-width/2
    inward=(np.arange(nd)+.5)*length/nd
    points=xy+transverse[:,None,None]*lateral-inward[None,:,None]*radial
    ij=np.floor((points-origin)/cell).astype(int)
    if np.any(ij[...,0]<0) or np.any(ij[...,0]>=h.shape[1]) or np.any(ij[...,1]<0) or np.any(ij[...,1]>=h.shape[0]):
        raise ValueError('cut_band_outside_observation')
    heights=h[ij[...,1],ij[...,0]]
    if not np.isfinite(heights).all() or np.any(count[ij[...,1],ij[...,0]]<1):
        raise ValueError('cut_band_unknown_observation')
    thickness=np.clip(heights-bottom,0,band_height)
    mean_width=thickness.mean(axis=1)
    weak=float(np.sort(mean_width)[:max(1,nw//3)].mean())
    volume=float(thickness.mean()*width*length)
    weak_equivalent=weak*width*length
    score=.5*(volume+weak_equivalent)
    return dict(version=VERSION,score_m3=score,observed_cut_band_volume_m3=volume,
                weakest_third_equivalent_volume_m3=weak_equivalent,
                width_band_mean_thickness_m=mean_width.tolist(),
                material_coverage_fraction=float(np.mean(thickness>=.15)),
                width_m=width,inward_length_m=length,bottom_z_m=bottom,
                below_entry_m=below_entry,vertical_band_height_m=band_height,
                sample_spacing_max_m=cell/2,
                scope='Observed geometric cut opportunity; not retained payload or fill prediction')


def target_mouth_markers(selected_cut, width=.60):
    """Fixed desired mouth span, perpendicular to the radial approach heading."""
    if not np.isfinite(width) or width<=0:
        raise ValueError('invalid_target_width')
    center=np.r_[np.asarray(selected_cut['entry_xy_m'],float),float(selected_cut['surface_z_m'])]
    if not np.isfinite(center).all():
        raise ValueError('invalid_target_height')
    _, lateral=target_basis(center[:2])
    offset=np.r_[lateral*width/2,0.]
    return dict(center=center,side_negative=center-offset,side_positive=center+offset,
                width_m=width,scope='fixed planned mouth span, not current bucket edges')

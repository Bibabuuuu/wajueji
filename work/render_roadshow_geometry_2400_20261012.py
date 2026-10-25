"""Render audited geometry display cache to screenshots and a short MP4.

No physics stepping, policy inference, or state modification.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path('/home/qunwang/wajueji')
RUN = ROOT / 'runs/roadshow-measurements-20261011/geometry-independent-20261012-v4'
CACHE = RUN / 'physics/display'
OUT = RUN / 'offline-analysis'
sys.path.insert(0, str(ROOT / 'src/v4'))
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def render(smoke):
    import pyglet.display.xlib as xlib
    if not hasattr(xlib.XlibScreen, 'is_primary'):
        xlib.XlibScreen.is_primary = False
    import newton
    import warp as wp
    import run_layered_47_sparse_manual_contact_episode as manual
    from soil_surface import SoilSurface

    review = json.loads((OUT / 'review.json').read_text())
    cache_report = json.loads((CACHE / 'reconstruction.json').read_text())
    assert review['status'] == 'passed' and cache_report['frames'] == 2400
    assert review['display_cache_hashes'] == cache_report['cache_sha256']
    for name in ('states.json', 'scene.json'):
        assert sha(CACHE / name) == cache_report['cache_sha256'][name]
    states = json.loads((CACHE / 'states.json').read_text())
    scene = json.loads((CACHE / 'scene.json').read_text())
    particles = np.load(CACHE / 'particles.npy', mmap_mode='r')
    assert len(states) == len(particles) == 2401
    display = json.loads((ROOT / 'configs/v4/layered-47-interactive-proxy-practice-v1.json').read_text())['display']
    width, height = display['width'], display['height']
    wp.init()
    wp.set_device('cuda:0')
    assembly, model, model_state, names, to_pose = manual.display_helpers._display_runtime(display)
    viewer = newton.viewer.ViewerGL(width=width, height=height, vsync=False, paused=True, headless=True)
    viewer.set_model(model)
    viewer.picking_enabled = False
    viewer.set_camera(wp.vec3(*display['camera_position_m']), pitch=display['camera_pitch_deg'],
                      yaw=display['camera_yaw_deg'])
    positions = wp.array(particles[0], dtype=wp.vec3, device='cuda:0')
    fixture = SimpleNamespace(state=SimpleNamespace(particle_q=positions))
    surface = SoilSurface(positions, particle_spacing_m=scene['particle_spacing_m'],
                          voxel_size_m=display['surface_voxel_m'])
    manual.display_helpers._prepare_rock_visuals(viewer, scene['records'], Path(scene['root']), 'cuda:0')
    selected = scene['selected_cut']
    face = selected.get('contact', selected)
    goal = dict(entry_xy_m=face.get('contact_xy_m', face.get('entry_xy_m')),
                surface_z_m=face.get('contact_surface_z_m', face.get('surface_z_m')))

    def frame_image(frame):
        positions.assign(particles[frame])
        state = {**states[frame], 'bucket_origin': states[frame]['bucket_origin_world_m']}
        manual.draw_with_selected_cut(viewer, model_state, fixture, state, scene['records'], assembly,
                                      names, to_pose, 'cuda:0', display, goal, frame, surface)
        image = viewer.get_frame(render_ui=False).numpy()
        assert image.shape == (height, width, 3)
        return image

    started = time.monotonic()
    try:
        for _ in range(3):
            image = frame_image(900)
        if smoke:
            destination = OUT / 'render-smoke-0900.png'
            Image.fromarray(image).save(destination)
            print(json.dumps(dict(status='smoke_passed', path=str(destination), sha256=sha(destination),
                                  wall_seconds=time.monotonic() - started)))
            return
        screenshots = OUT / 'screenshots'
        assert not screenshots.exists() and not (OUT / 'geometry-2400-2x.mp4').exists()
        screenshots.mkdir()
        keyframes = (0, 73, 394, 900, 1831, 2400)
        for frame in keyframes:
            Image.fromarray(frame_image(frame)).save(screenshots / f'geometry-{frame:04d}.png')
        movie = OUT / 'geometry-2400-2x.mp4'
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
                   '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', str(movie)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            assert process.stdin is not None
            for frame in range(0, 2401, 4):
                process.stdin.write(frame_image(frame).tobytes())
            process.stdin.close()
            assert process.wait() == 0, 'ffmpeg_failed'
        files = [*sorted(screenshots.glob('*.png')), movie]
        result = dict(status='rendered', source_run=str(RUN), playback_physics_steps=0,
                      controller='geometry_feedback_without_neural_weights',
                      keyframes=list(keyframes), movie=dict(path=str(movie), frames=601, fps=30,
                                                             simulated_frames_per_video_frame=4,
                                                             playback_speed=2.0),
                      sources=dict(review_sha256=sha(OUT / 'review.json'),
                                   cache_report_sha256=sha(CACHE / 'reconstruction.json')),
                      outputs=[dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for path in files],
                      wall_seconds=time.monotonic() - started)
        (OUT / 'render-review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(dict(status='rendered', movie=str(movie), screenshots=len(keyframes),
                              wall_seconds=result['wall_seconds'])))
    finally:
        viewer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    render(parser.parse_args().smoke)

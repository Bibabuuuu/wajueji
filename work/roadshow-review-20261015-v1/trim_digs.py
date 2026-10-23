from pathlib import Path
import subprocess,json
R=Path('/home/qunwang/wajueji');O=R/'runs/roadshow-review-20261015-v1/video'
src=R/'runs/roadshow-two-scoop-media-20261015-v1/video/two-scoop-full-1x.mp4'
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
style=f'fontfile={font}:fontsize=38:fontcolor=white:x=30:y=26'
def segment(label,start,end):
 return f'trim=start_frame={start}:end_frame={end},setpts=(PTS-STARTPTS)/2,drawbox=x=0:y=0:w=iw:h=116:color=0x102535:t=fill,drawbox=x=0:y=1025:w=iw:h=55:color=0x102535:t=fill,drawtext={style}:text={label}'
f=f'[0:v]split=2[a][b];[a]{segment("首铲挖掘与保载 · 2倍速",2,1931)}[first];[b]{segment("同一土堆 · 第二铲挖掘与保载 · 2倍速",7687,9448)}[second];[first][second]concat=n=2:v=1:a=0,fps=30[out]'
out=O/'two-digs-highlights-2x.mp4'
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(src),'-filter_complex',f,'-map','[out]','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-ss','13','-i',str(out),'-frames:v','1',str(O/'two-digs-poster.png')],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_streams','-of','json',str(out)]))
(O/'dig-trim-report.json').write_text(json.dumps(dict(status='completed',source=str(src),source_video_fps=30,source_intervals_frames=[[2,1931],[7687,9448]],source_workflow_intervals=[[4,3860],[15374,18894]],output_speed=2,edited_header=True,source_motion_unchanged=True,physics_steps=0,scope='two real digging and hold segments; intervening transport omitted; not a rerun of full two-scoop chain',stream=probe['streams'][0]),ensure_ascii=False,indent=2))
print('TRIM_COMPLETED',probe['streams'][0]['duration'])

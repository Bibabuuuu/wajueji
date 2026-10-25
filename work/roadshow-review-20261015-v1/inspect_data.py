from pathlib import Path
import json
R=Path('/home/qunwang/wajueji')
for p in sorted(R.glob('work/*stage03*')):print('WORK',p)
for p in sorted(R.glob('runs/*stage03*')):print('RUN',p)
for p in sorted(R.glob('runs/*three*')):print('THREE',p)
for p in sorted(R.glob('work/*three*')):print('THREEWORK',p)
for p in sorted(R.glob('runs/*d165*')):print('D165',p)

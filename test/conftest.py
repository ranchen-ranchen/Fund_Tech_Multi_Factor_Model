import sys
from pathlib import Path

# 把项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "strategy"))

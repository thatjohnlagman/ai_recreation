import os
import re
from pathlib import Path

# Fix imports in python files
def fix_imports():
    demo_dir = Path(__file__).parent
    for root, _, files in os.walk(demo_dir):
        for file in files:
            if file.endswith('.py') and file != 'build_package.py':
                path = Path(root) / file
                with open(path, 'r') as f:
                    content = f.read()
                
                content = content.replace('recall_aware_ids.', '')
                
                # Fix PROJECT_ROOT in config.py
                if file == 'config.py':
                    content = re.sub(
                        r'PROJECT_ROOT = .*',
                        'PROJECT_ROOT = Path(__file__).parents[1]',
                        content
                    )
                
                # In base.py of attacks or defenses, might need to fix paths or other PROJECT_ROOT references
                content = re.sub(r'PROJECT_ROOT\s*/\s*"artifacts/preprocessors"', 'PROJECT_ROOT / "model"', content)
                content = re.sub(r'PROJECT_ROOT\s*/\s*"artifacts/models"', 'PROJECT_ROOT / "model"', content)

                with open(path, 'w') as f:
                    f.write(content)

fix_imports()

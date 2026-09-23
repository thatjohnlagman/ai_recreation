import re

def patch_file(fname):
    with open(fname, "r") as f:
        text = f.read()
    
    # Add git commit output
    if "import subprocess" not in text:
        text = "import subprocess\n" + text
        
    head_code = """    command = " ".join(sys.argv)
    head_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    
    print(f"--- [{stage}]"""
    text = re.sub(r'    command = " ".join\(sys\.argv\)\s+print\(f"--- \[\{stage\}\]', head_code, text)
    
    head_print = """    print(f"Command: {command}")
    print(f"Commit: {head_commit}")"""
    text = re.sub(r'    print\(f"Command: \{command\}"\)', head_print, text)
    
    with open(fname, "w") as f:
        f.write(text)

patch_file("scripts/verify_immutability.py")
patch_file("scripts/verify_protected_caches.py")

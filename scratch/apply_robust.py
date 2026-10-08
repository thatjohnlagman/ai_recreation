import re

with open("scratch/safe_results.js", "r") as f:
    content = f.read()

# Get arrays from canonical_values.js
with open("scratch/canonical_values.js", "r") as f:
    canonical = f.read()
    
summary_match = re.search(r'(const STATIC_BENCHMARK_SUMMARY = \[.*?\];)', canonical, re.DOTALL)
matrix_match = re.search(r'(const STATIC_C1_C7_MATRIX = \[.*?\];)', canonical, re.DOTALL)

summary_str = summary_match.group(1)
matrix_str = matrix_match.group(1)

# Now manually parse and replace in content safely by tracking brackets
def replace_array(content, array_name, replacement):
    idx = content.find(f"const {array_name} = [")
    if idx == -1: return content
    
    start_bracket = content.find("[", idx)
    end_idx = start_bracket + 1
    depth = 1
    
    while end_idx < len(content) and depth > 0:
        if content[end_idx] == '[': depth += 1
        elif content[end_idx] == ']': depth -= 1
        end_idx += 1
        
    semicolon_idx = content.find(";", end_idx)
    if semicolon_idx != -1 and semicolon_idx - end_idx < 10:
        end_idx = semicolon_idx + 1
        
    return content[:idx] + replacement + content[end_idx:]

new_content = replace_array(content, "STATIC_BENCHMARK_SUMMARY", summary_str)
new_content = replace_array(new_content, "STATIC_C1_C7_MATRIX", matrix_str)

with open("frontend/results.js", "w") as f:
    f.write(new_content)

print("Restored safely!")

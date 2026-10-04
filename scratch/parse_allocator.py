import re

with open('CUDACachingAllocator.cpp', 'r') as f:
    content = f.read()

def extract_block(pattern, text, max_lines=100):
    match = re.search(pattern, text)
    if not match:
        return ""
    start_idx = match.start()
    lines = text[start_idx:].split('\n')
    brace_count = 0
    in_block = False
    result = []
    for line in lines:
        if '{' in line:
            brace_count += line.count('{')
            in_block = True
        if '}' in line:
            brace_count -= line.count('}')
        
        result.append(line)
        if in_block and brace_count == 0:
            break
        if len(result) > max_lines:
            break
    return '\n'.join(result)

print("--- struct BlockPool ---")
print(extract_block(r'struct BlockPool ', content, max_lines=40))
print("\n--- struct DeviceCachingAllocator ---")
print(extract_block(r'class CachingAllocator ', content, max_lines=100))


import re

with open('CUDACachingAllocator.cpp', 'r') as f:
    text = f.read()

def extract_method(pattern, text):
    match = re.search(pattern, text)
    if not match:
        return "Not found"
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
    return '\n'.join(result)

print("--- malloc ---")
print(extract_method(r'Block\* malloc\(int device, size_t original_size, cudaStream_t stream\)', text)[:1500])
print("\n--- free ---")
print(extract_method(r'void free\(Block\* block\)', text)[:1500])


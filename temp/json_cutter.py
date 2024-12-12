import json
old_json_path = '/home1/hxl/disk/EAGLE/dataset/Eagle-1.8M/eagle-1-sft-1_8M.json'
# new_json_path = '/home1/hxl/disk/EAGLE/dataset/Eagle-1.8M/eagle-100.json'
new_json_path = '/home1/hxl/disk/EAGLE/dataset/llava_pretrain/blip_laion_cc_sbu_558k.json'
json_path_10 = '/home1/hxl/disk/EAGLE/dataset/llava_pretrain/blip_laion_cc_sbu_10.json'
with open(new_json_path, 'r') as f:
    data = json.load(f)
    
# Randomly select sample_size entries from the data
new_data = data[:10]
    
# Write the subsampled data to the output file
with open(json_path_10, 'w') as f:
    json.dump(new_data, f)
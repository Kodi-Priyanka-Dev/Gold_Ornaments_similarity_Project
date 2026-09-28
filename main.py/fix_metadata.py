import json

with open('metadata.json', 'r') as f:
    data = json.load(f)

count = 0
for path, meta in data.items():
    lower_path = path.lower().replace('\\', '/')
    if '/necklace' in lower_path:
        meta['category'] = 'Necklace'
        count += 1
    elif '/ear' in lower_path:
        meta['category'] = 'Ear Rings'

print(f"Fixed {count} necklaces!")

with open('metadata.json', 'w') as f:
    json.dump(data, f, indent=4)

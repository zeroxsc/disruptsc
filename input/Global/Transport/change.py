import json



# 读取GeoJSON文件
with open('output_file.geojson', 'r') as f:
    geojson_data = json.load(f)

seen_iso3 = set()

# 逐个处理 features 中的每个 'iso3' 列
for feature in geojson_data['features']:
    iso3_value = feature['properties'].get('iso3', None)

    # 如果 iso3 已经出现过，设置为 null
    if iso3_value in seen_iso3:
        feature['properties']['iso3'] = None
    elif iso3_value:
        # 如果是第一次出现该 iso3，记录下来
        seen_iso3.add(iso3_value)

# 保存处理后的 GeoJSON 数据
with open('processed_geojson_file3.geojson', 'w') as f:
    json.dump(geojson_data, f, indent=2)
import pickle
import pandas as pd
from network import route
import geopandas as gpd
from shapely.geometry import Point

from network.route import Route

# 读取 pickle 文件
with open(r'C:\Users\86156\Desktop\disruptsc-dade\disruptsc-dade\tmp\supply_chain1', 'rb') as f:
    data = pickle.load(f)
with open(r'C:\Users\86156\Desktop\disruptsc-dade\disruptsc-dade\tmp\supply_chain2', 'rb') as f:
    data1 = pickle.load(f)
with open(r'C:\Users\86156\Desktop\disruptsc-dade\disruptsc-dade\tmp\supply_chain5', 'rb') as f:
    data2 = pickle.load(f)

pd.set_option('display.max_columns', None)
# 现在可以使用加载的数据
print(data.keys())  # 查看数据内容
print(data.keys())  # 查看数据内容
scnetwork=data2['supply_chain_network']
a=scnetwork.adjacency()
# print(a)
# print('图中所有的边', scnetwork.edges())
count1=0
count2=0
for source, target, attrs in scnetwork.edges(data=True):
    print(f"源节点: {source}")
    print(f"目标节点: {target}")
    # 打印边的属性
    print(f"属性: {attrs}")
    # 获取并查看 CommercialLink 对象
    commercial_link_obj = attrs.get('object', None)
    attributes = vars(commercial_link_obj)

    for attribute, value in attributes.items():
        if attribute == 'route':  # 检查是否是 'route' 属性
            if type(value)== list and len(value)>0:
                count1+=1
                print(value)
        else:
            print(f"{attribute} = {value}")
    print("-" * 40)
print(count1)

import csv
# 定义CSV文件的字段名
fieldnames = [
    'source_id',
    'source_agent_type',
    'source_sector',
    'source_long',
    'source_lat',
    'target_id',
    'target_agent_type',
    'target_sector',
    'target_long',
    'target_lat',
    'attrs'
]

# 打开CSV文件以写入模式
with open('output2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入表头
    writer.writeheader()

    for source, target, attrs in scnetwork.edges(data=True):
        # 假设source和target是节点对象，若是节点ID，请根据实际情况获取对象
        # 如果source和target是ID，可以通过scnetwork.nodes[source]['object']获取对象属性
        source_obj = source  # 根据实际情况调整
        target_obj = target  # 根据实际情况调整

        # 创建一行数据
        row = {
            'source_id': source,  # 如果source是对象，可以使用source.id或其他唯一标识
            'source_agent_type': getattr(source_obj, 'agent_type', ''),
            'source_sector': getattr(source_obj, 'sector', ''),
            'source_long': getattr(source_obj, 'long', ''),
            'source_lat': getattr(source_obj, 'lat', ''),
            'target_id': target,  # 如果target是对象，可以使用target.id或其他唯一标识
            'target_agent_type': getattr(target_obj, 'agent_type', ''),
            'target_sector': getattr(target_obj, 'sector', ''),
            'target_long': getattr(target_obj, 'long', ''),
            'target_lat': getattr(target_obj, 'lat', ''),
            'attrs': attrs
        }

        # 写入当前行到CSV文件
        writer.writerow(row)

print(type(a))
# for k in a:
#     print (k)

# print(scnetwork)
# print(scnetwork.nodes)
# count=0
# for node in scnetwork.nodes:
#     count+=1
#     print(node.id_str())
edges=scnetwork.generate_edge_list()
nb1=edges['source_id'].nunique()  #5054
nb2=edges['source_od_point'].nunique()  #187
nb3=edges['target_od_point'].nunique()  #187
nb4=edges['target_id'].nunique()
print("source_id",nb1)
print("source_od_point",nb2)
# print("scnode",count)
print("target_id",nb4)
print("target_od_point",nb3)

print(edges)
edges.to_csv("a2.csv")

print("source_type",edges['source_type'].unique())
source_type_counts = edges['source_type'].value_counts()
print("source_type 'country' count:", source_type_counts.get('country', 0))
print("source_type 'firm' count:", source_type_counts.get('firm', 0))

target_type_counts = edges['target_type'].value_counts()
print("target_type 'country' count:", target_type_counts.get('country', 0))
print("target_type 'firm' count:", target_type_counts.get('firm', 0))
print("target_type 'household' count:", target_type_counts.get('household', 0))
print("target_type",edges['target_type'].unique())


#
# with open(r'E:\Zero\disruptsc-dade\disruptsc-dade\tmp\transport_network_pickle', 'rb') as t:
#     data2 = pickle.load(t)
# trans=data2['transport_network']
# nodes=data2['transport_nodes']
# edges=data2['transport_edges']
# country=nodes["iso3"]
# print(country)
# nbofc=country.nunique()
# print("country",nbofc)
# country_list=country.unique()

# print(country_list)
# print(len(country_list))
# print(nodes)
# print(edges)
# print(data2.keys())
# print(trans)
# table=edges

# 将 edges 按 source 和 target 分为两个表
# edges_source = edges[['source_id', 'source_od_point', 'source_type']].copy()
# edges_source['geometry'] = edges_source['source_od_point'].map(
#     nodes.set_index('id')['geometry'])
#
# edges_source_gdf = gpd.GeoDataFrame(
#     edges_source,
#     geometry='geometry'
# )

# 导出到 GeoJSON 文件
# edges_source_gdf.to_file('edges_source.geojson', driver='GeoJSON')
#
#
# edges_target = edges[['target_id', 'target_od_point', 'target_type']].copy()
# edges_target['geometry'] = edges_target['target_od_point'].map(
#     nodes.set_index('id')['geometry'])


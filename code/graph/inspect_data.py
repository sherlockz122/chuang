# 环节一/00: 检查 BBBike 北京提取包的结构与字段（跑通后即可删或留作文档）
import sys, zipfile, os
sys.stdout.reconfigure(encoding='utf-8')
import geopandas as gpd

SRC = r'D:\大创项目实存\new_route\data\osm_beijing\Beijing.osm.shp.zip'

with zipfile.ZipFile(SRC) as z:
    names = [n for n in z.namelist() if n.lower().endswith('.shp')]
    print('shp 文件列表:')
    for n in names:
        print('  ', n)

    for n in names:
        # 解压到临时目录后读取（shp 需要同目录的 dbf/shx）
        tmp = os.path.join(os.environ['TEMP'], 'bbbike_inspect')
        os.makedirs(tmp, exist_ok=True)
        for member in z.namelist():
            if member.split('/')[-1].startswith(n.split('/')[-1].split('.')[0]):
                z.extract(member, tmp)
        shp = os.path.join(tmp, n)
        gdf = gpd.read_file(shp)
        print(f'\n===== {n} ===== 行数 {len(gdf)}')
        print('字段:', list(gdf.columns))
        print('几何类型:', gdf.geom_type.unique()[:5])
        if len(gdf):
            row = gdf.iloc[0]
            print('示例行:')
            for c in gdf.columns:
                v = str(row[c])
                if len(v) > 120:
                    v = v[:120] + '...'
                print(f'    {c}: {v}')

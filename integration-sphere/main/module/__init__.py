#MCパラメーター読み込み

#.xlsxと読み込むシートの明示
wb_1 = px.load_workbook((Parameter_File))
sheet = wb_1['parameter']

#値の読み込み
data_per_file = int(sheet.cell(3, 1).value)
num_photons   = int(sheet.cell(5, 1).value)
dz            = float(sheet.cell(6, 1).value)
dr            = float(sheet.cell(6, 2).value)
num_dz        = int(sheet.cell(7, 1).value)
num_dr        = int(sheet.cell(7, 2).value)
num_da        = int(sheet.cell(7, 3).value)

num_layers    = int(sheet.cell(10, 1).value)
n_above       = float(sheet.cell(12, 1).value)
mua_layer_1   = float(sheet.cell(13, 2).value)
mua_layer_3   = float(sheet.cell(15, 2).value)
mus_layer_1   = float(sheet.cell(13, 3).value)
mus_layer_3   = float(sheet.cell(15, 3).value)
n_layer_1     = float(sheet.cell(13, 1).value)
n_layer_2     = float(sheet.cell(14, 1).value)
n_layer_3     = float(sheet.cell(15, 1).value)
g_layer_1     = float(sheet.cell(13, 4).value)
g_layer_2     = float(sheet.cell(14, 4).value)
g_layer_3     = float(sheet.cell(15, 4).value)
d_layer_1     = float(sheet.cell(13, 5).value)
d_layer_2     = float(sheet.cell(14, 5).value)
d_layer_3     = float(sheet.cell(15, 5).value)
n_below       = float(sheet.cell(16, 1).value)

wb_1.close()


#ランクの定義
rank_list    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
interval_mua = {'rank1': 1, 'rank2': 0.1, 'rank3': 0.05, 'rank4': 0.02, 'rank5': 0.01, 'rank6': 0.005, 'rank7': 0.002, 'rank8': 0.001, 'rank9': 0.0005, 'rank10': 0.0002, 'rank11': 0.0001, 'rank12': 0.00005}
interval_mus = {'rank1': 2, 'rank2': 1, 'rank3': 0.5, 'rank4': 0.2, 'rank5': 0.1, 'rank6': 0.05, 'rank7': 0.02, 'rank8': 0.01, 'rank9': 0.005, 'rank10': 0.002, 'rank11': 0.001, 'rank12': 0.0005}


#日付の定義
if date == 'True':
    d_today = datetime.date.today()
    item    = str(d_today).rstrip('\n').split('-')
    string    = item[0][2:4] + item[1] + item[2]
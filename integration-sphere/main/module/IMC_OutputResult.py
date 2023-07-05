#[定義] 途中経過の.xlsx出力、図作成など

'''
# 1  そのランクでのmua, mus、Rd, Ttの誤差、参照MCデータ
# 2  次ランクでのMCシミュレーションのペア等の表を出力
# 3  そのランクでのmua, musのスペクトル、誤差情報の図を出力
# 4  最終的なmuaのスペクトル
# 5  最終的なmusのスペクトル
'''


class Output_result():
    # 1
    def Create_table(self):
        wb_2 = px.Workbook()

        #Sheet['Cal']
        sheet_1 = wb_2.active

        #title
        sheet_1.title = 'Cal'
        sheet_1.cell(1, 1).value  = '累計MCデータ数'
        sheet_1.cell(3, 1).value  = 'Exデータ数'
        sheet_1.cell(5, 1).value  = '0.5%以上の数'
        sheet_1.cell(8, 1).value  = 'Rank'
        sheet_1.cell(10, 1).value = '分解能'
        sheet_1.cell(11, 1).value = 'mua'
        sheet_1.cell(13, 1).value = 'mus'
        sheet_1.cell(1, 2).value  = 'Wavelength [nm]'
        sheet_1.cell(1, 3).value  = 'Rd'
        sheet_1.cell(1, 4).value  = 'Tt'
        sheet_1.cell(1, 5).value  = 'mua'
        sheet_1.cell(1, 6).value  = 'mus'
        sheet_1.cell(1, 7).value  = 'Rd_cal'
        sheet_1.cell(1, 8).value  = 'Tt_cal'
        sheet_1.cell(1, 9).value  = 'Rd 誤差 [%]'
        sheet_1.cell(1, 10).value = 'Tt 誤差 [%]'

        #value
        sheet_1.cell(2, 1).value  = len(simulated_data)
        sheet_1.cell(4, 1).value  = num_measured_data
        sheet_1.cell(6, 1).value  = data_pair.shape[0]
        sheet_1.cell(9, 1).value  = rank
        sheet_1.cell(12, 1).value = interval_mua['rank{:}'.format(rank)]
        sheet_1.cell(14, 1).value = interval_mus['rank{:}'.format(rank)]

        #data
        for i in range(RESULT.shape[1]):
            for j in range(num_measured_data):
                sheet_1.cell(j+2, i+2).value = RESULT[j, i]

        #Sheet['Data']
        sheet_2 = wb_2.create_sheet(title='Data')
        wb_2.active = wb_2.sheetnames.index('Data')

        sheet_2.cell(1, 1).value = 'mua'
        sheet_2.cell(1, 2).value = 'mus'
        sheet_2.cell(1, 3).value = 'Rd_sim'
        sheet_2.cell(1, 4).value = 'Tt_sim'
        sheet_2.cell(1, 6).value = '累計データ数'
        sheet_2.cell(2, 6).value = len(simulated_data)
        sheet_2.cell(3, 6).value = 'データ数（現ランク）'
        sheet_2.cell(4, 6).value = len(tmp_simulated_data)

        for i in range(0, 3):
            for j in range(len(simulated_data)):
                sheet_2.cell(j+2, i+1).value = simulated_data[j, i]

        if date == 'True':
            wb_2.save('{:}/rank{:}/databese_rank{:}_{:}.xlsx'.format(log_path, rank, rank, string))
        else:
            wb_2.save('{:}/rank{:}/databese_rank{:}.xlsx'.format(log_path, rank, rank))
        wb_2.close()


    # 2
    def Create_table_next(self):
        #Sheet['Next']
        if date == 'True':
            wb_2 = px.load_workbook('{:}/rank{:}/databese_rank{:}_{:}.xlsx'.format(log_path, rank, rank, string))
        else:
            wb_2 = px.load_workbook('{:}/rank{:}/databese_rank{:}.xlsx'.format(log_path, rank, rank))

        sheet_3 = wb_2.create_sheet(title='Next')
        wb_2.active = wb_2.sheetnames.index('Next')

        sheet_3.cell(1, 4).value  = 'データ数（次ランク）'
        sheet_3.cell(2, 4).value  = len(MC_pair)
        sheet_3.cell(1, 1).value  = 'mua_next'
        sheet_3.cell(1, 2).value  = 'mus_next'
        sheet_3.cell(4, 4).value  = '現Rank'
        sheet_3.cell(5, 4).value  = rank
        sheet_3.cell(4, 5).value  = 'mua間隔'
        sheet_3.cell(5, 5).value  = interval_mua['rank{:}'.format(rank)]
        sheet_3.cell(6, 5).value  = 'mus間隔'
        sheet_3.cell(7, 5).value  = interval_mus['rank{:}'.format(rank)]
        sheet_3.cell(8, 4).value  = '次ランク'
        sheet_3.cell(9, 4).value  = rank+1
        sheet_3.cell(8, 5).value  = 'mua間隔'
        sheet_3.cell(9, 5).value  = interval_mua['rank{:}'.format(rank+1)]
        sheet_3.cell(10, 5).value = 'mus間隔'
        sheet_3.cell(11, 5).value = interval_mus['rank{:}'.format(rank+1)]

        for i in range(0, MC_pair.shape[1]):
            for j in range(len(MC_pair)):
                sheet_3.cell(j+2, i+1).value = MC_pair[j, i]

        wb_2.active = wb_2.sheetnames.index('Cal')

        if date == 'True':
            wb_2.save('{:}/rank{:}/databese_rank{:}_{:}.xlsx'.format(log_path, rank, rank, string))
        else:
            wb_2.save('{:}/rank{:}/databese_rank{:}.xlsx'.format(log_path, rank, rank))
        wb_2.close()


    # 3
    def Create_figures(self):
        plt.rcParams['font.size'] = 16

        #サブプロットの設定
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        plt.subplots_adjust(wspace=0.3, hspace=0.2)

        #軸ラベル
        for i in range(3):
            for j in range(2):
                axes[i][j].set_xlabel('Wavelength [nm]')
        axes[0][0].set_ylabel(r"$\mu_{a}$ " + "[mm" + r"$^{-1}]$")
        axes[0][1].set_ylabel(r"$\mu_{s}$ " + "[mm" + r"$^{-1}]$")
        axes[1][0].set_ylabel('Diffuse reflectance [-]')
        axes[1][1].set_ylabel('Total transmittance [-]')
        axes[2][0].set_ylabel('Relative error (' + r'$R_{d}$' + ') [%]')
        axes[2][1].set_ylabel('Relative error (' + r'$T_{t}$' + ') [%]')

        #プロット
        axes[0][0].scatter(wavelength, RESULT[:, 3], s=1)
        axes[0][1].scatter(wavelength, RESULT[:, 4], s=1)
        axes[1][0].scatter(wavelength, RESULT[:, 1], s=1, c='blue', label='Measured')
        axes[1][0].scatter(wavelength, RESULT[:, 5], s=1, c='red', marker='x', label='Calculated')
        axes[1][1].scatter(wavelength, RESULT[:, 2], s=1, c='blue', label='Measured')
        axes[1][1].scatter(wavelength, RESULT[:, 6], s=1, c='red', marker='x',label='Calculated')
        axes[2][0].scatter(wavelength, RESULT[:, 7], s=1, c='black')
        axes[2][1].scatter(wavelength, RESULT[:, 8], s=1, c='black')

        #凡例
        axes[1][0].legend(bbox_to_anchor=(0.98, 0.02), loc='lower right', borderaxespad=0, fontsize=12)
        axes[1][1].legend(bbox_to_anchor=(0.98, 0.02), loc='lower right', borderaxespad=0, fontsize=12)

        #グラフの閾値の線
        axes[2][0].axhline(0.5, 0, 1, color='red', linestyle='dashed', linewidth=2)
        axes[2][1].axhline(0.5, 0, 1, color='red', linestyle='dashed', linewidth=2)

        #保存
        if date == 'True':
            fig.savefig('{:}/rank{:}/IMG_rank{:}_{:}.jpeg'.format(log_path, rank, rank, string), dpi=600, bbox_inches='tight')
        else:
            fig.savefig('{:}/rank{:}/IMG_rank{:}.jpeg'.format(log_path, rank, rank), dpi=600, bbox_inches='tight')


    # 4
    def Final_spectra(self):
        plt.rcParams['font.size'] = 16
        
        #mua
        mua_fig = plt.figure(figsize=(6, 4))
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(r"$\mu_{a}$ " + "[mm" + r"$^{-1}]$")
        plt.scatter(wavelength, RESULT[:, 3], s=1)

        mua_fig.savefig('{:}/mua.jpeg'.format(log_path), dpi=600, bbox_inches='tight')
        mua_fig.savefig('{:}/mua.tif'.format(log_path), dpi=250, bbox_inches='tight')

        #mus
        mus_fig = plt.figure(figsize=(6, 4))
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(r"$\mu_{s}$ " + "[mm" + r"$^{-1}]$")
        plt.scatter(wavelength, RESULT[:, 4], s=1)

        mus_fig.savefig('{:}/mus.jpeg'.format(log_path), dpi=600, bbox_inches='tight')
        mus_fig.savefig('{:}/mus.tif'.format(log_path), dpi=250, bbox_inches='tight')

        #reduced_mus
        reduced_mus_fig = plt.figure(figsize=(6, 4))
        plt.xlabel('Wavelength [nm]')
        plt.ylabel(r"$\mu_{s}^{'}$ " + "[mm" + r"$^{-1}]$")
        plt.scatter(wavelength, reduced_mus, s=1)

        reduced_mus_fig.savefig('{:}/reduced_mus.jpeg'.format(log_path), dpi=600, bbox_inches='tight')
        reduced_mus_fig.savefig('{:}/reduced_mus.tif'.format(log_path), dpi=250, bbox_inches='tight')
    
    
    # 5
    def Average_spectra(self, data, label):
        plt.rcParams['font.size'] = 16

        wavelength_output = np.split(wavelength, num_spectra)
        data =  data.reshape(num_spectra, -1)
        ave = np.mean(data, axis=0).round(5)
        std = np.std(data, axis=0).round(5)

        ave_fig = plt.figure(figsize=(6, 4))
        plt.xlabel('Wavelength [nm]')

         
        if label == 'mua':
            plt.ylabel(r"$\mu_{a}$ " + "[mm" + r"$^{-1}]$")
        elif label == 'mus':
            plt.ylabel(r"$\mu_{s}$ " + "[mm" + r"$^{-1}]$")
        else:
            plt.ylabel(r"$\mu_{s}^{'}$ " + "[mm" + r"$^{-1}]$")
        
        plt.errorbar(wavelength_output[0], ave, yerr=std,
                     markersize=2, ecolor='powderblue', capsize=0.1, fmt='o')
        
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        plt.text(x_min+(x_max-x_min)*0.95, y_min+(y_max-y_min)*0.95,
                 'N = {:}'.format(num_spectra),
                horizontalalignment='right', verticalalignment='top')
        
        ave_fig.savefig('{:}/ave_{:}.jpeg'.format(log_path, label), dpi=600, bbox_inches='tight')
        ave_fig.savefig('{:}/ave_{:}.tif'.format(log_path, label), dpi=250, bbox_inches='tight')
        

    # 6
    def Csv_output(self):
        reduced_mus = RESULT[:, 4]*(1-g_layer_2)
        output_data = np.c_[wavelength, RESULT[:, 3], RESULT[:, 4], reduced_mus, RESULT[:, 1], RESULT[:, 2]]
        output_data_df = pd.DataFrame(output_data, columns=["wavelength", "mua [mm-1]", "mus [mm-1]", "mus' [mm-1]", "Rd", "Tt"])
        output_data_df.to_csv('{:}/IMC_result.csv'.format(log_path), index=False)
        


        


        

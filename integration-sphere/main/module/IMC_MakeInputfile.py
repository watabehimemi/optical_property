#テキストの定義

from fileinput import filename

class WriteText():

    def Write_Title(self, num):
        title = '### Template of input files for Monte Carlo simuation (mcml).\n\
# Anything in a line after # is ignored as comments.\n\
# Space lines are also ignored.\n\
# Lengths are in mm, mua and mus are in 1/mm.\n\n\
1.0        # file version\n\
'+ str(num) +'     # number of runs\n\n'
        return title


    def Write_MainText(self, num):
        maintext = '### SPECIFY DATA FOR RUN' + str(Totalcounter) + '\n\
data' +str(num+1)+'.mco                      # output file name, ASCII.\n\
'+ str(num_photons) +'                        # No. of photons\n\
'+ str(dz) + '  ' + str(dr) +'                     # dz, dr [mm]\n\
'+ str(num_dz) + '  ' + str(num_dr) + '  ' + str(num_da) +'                  # No. of dz, dr, da.\n\n\
'+ str(num_layers) +'                              # Number of layers\n\
#n mua mus g d                 # One line for each layer\n\
'+ str(n_above) +'                            # n for medium above\n\
'+ str(n_layer_1) + '  ' + str(mua_layer_1) + '  ' + str(mus_layer_1) + '  ' + str(g_layer_1) + '  ' + str(d_layer_1) +'      # layer 1\n\
'+ str(n_layer_2) + '  ' + str('{:.5f}'.format(MC_pair[num,0]))+ '  ' + str('{:.5f}'.format(MC_pair[num,1]))+ '  ' + str(g_layer_2) + '  ' + str(d_layer_2) + '   # layer 2\n\
'+ str(n_layer_3) + '  ' + str(mua_layer_3) + '  ' + str(mus_layer_3) + '  ' + str(g_layer_3) + '  ' + str(d_layer_3) +'      # layer 3\n\
'+ str(n_below) +'                            # n for medium below\n\n\n'
        return maintext


#テキスト作成

if __name__ == '__main__':
    
    instance = WriteText()

    Totalcounter = 0
    Filecounter = 0
    
    for i in range(0, num_MC_data//data_per_file+1):

        #data_per_fileの個数ごとに出力ファイルを変更する
        if Totalcounter % data_per_file == 0:
            Filecounter += 1
            num_calculated = data_per_file*(Filecounter-1)

            if num_MC_data - Totalcounter >= data_per_file:
                writtenNum = data_per_file

            elif num_MC_data - Totalcounter < data_per_file:
                writtenNum = num_MC_data - Totalcounter
                
            if date == 'True':
                file_name = '{:}/rank{:}/Inputfile_rank{:}_{:}_{:}.mci'.format(log_path, rank, rank, string, Filecounter)
            else:
                file_name = '{:}/rank{:}/Inputfile_rank{:}_{:}.mci'.format(log_path, rank, rank, Filecounter) 

            with open(file_name, mode='w') as f:
                f.write(instance.Write_Title(writtenNum))

        with open(file_name, mode='a') as f:
            for j in range(num_calculated, num_calculated + writtenNum):
                Totalcounter += 1
                f.write(instance.Write_MainText(j))

    print('rank{:} // Inputfile created'.format(rank))
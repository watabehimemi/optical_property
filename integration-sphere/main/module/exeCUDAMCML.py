
'''
回らないことがよくあるから、タイムアウトを用いた複雑なコードを書いている
'''

files = glob.glob('{:}/rank{:}/Inputfile_*.mci'.format(log_path, rank))
files = natsorted(files)
num_files = len(files)

#タイムアウトの設定
# @timeout(timeout_sec)

#CUDA処理
def CUDAMCML(file):

    #入力ファイル名
    input_name  = file

    #出力ファイル名
    number      = re.findall(r'\d+', file)[-1]

    if date == 'True':
        output_name = '{:}/rank{:}/CUDA_result_rank{:}_{:}_{:}.txt'.format(log_path, rank, rank, string, number)    #ここ変更するなら、上の「ファイルの番号部分を取り出す」も変わるかも
    else:
        output_name = '{:}/rank{:}/CUDA_result_rank{:}_{:}.txt'.format(log_path, rank, rank, number)

    with subprocess.Popen(['{:}/CUDAMCML'.format(CUDA_path), input_name], shell=False) as Simulation:
        try:
            Simulation.wait()
        except:                                         #設定したtimeoutが生じた場合
            Simulation.terminate()
            time.sleep(2)
            os.remove('results.txt')
            time.sleep(5)
            get_ipython().system('make clean')
            get_ipython().system('make')
        else:                                           #timeoutする前に回った場合
            del Simulation
            Result = subprocess.Popen(['mv', 'results.txt', output_name], shell=False)
            Result.wait()
            del Result
            time.sleep(2)


if __name__ == '__main__':
    input_files = files

    if os.path.isfile('{:}/CUDAMCML'.format(CUDA_path)) == True:
        get_ipython().system('make clean')
        time.sleep(1)
        get_ipython().system('make')
    else:
        get_ipython().system('make')

    #全部のファイル回りきるまで繰り返す
    while len(input_files) != 0:
        for file in input_files:
            CUDAMCML(file)
            time.sleep(20)                                  #少し時間を置いた方がよく回る気がする

        output_files = glob.glob('{:}/rank{:}/CUDA_result_*.txt'.format(log_path, rank))
        output_files = natsorted(output_files)

        #ファイルの番号部分を取り出す
        input_num  = [re.findall(r'\d+', files[i])[-1] for i in range(len(files))]
        output_num = [re.findall(r'\d+', output_files[j])[-1] for j in range(len(output_files))]

        #番号の共通してないところはシミュレーションが回りきってないところなので抽出
        lst = sorted(list(set(input_num) ^ set(output_num)))

        input_files = [files[int(num)-1] for num in lst]
        print('rank{:} // CUDAMCML {:}/{:} finished'.format(rank, num_files-len(input_files), num_files))

    print('rank{:} // CUDAMCML all finished'.format(rank))

#[定義] rankの移動

'''
#1 そのrankでmua, musの最適値を探索する
#2 次ランクでMCの入力として使うmua, musを決定する
'''

class MoveRank():
    # 1
    def Get_Possible_Values(self, measured, simulated):

        pair = []
        results = []

        simulated_R = simulated[:, 2]
        simulated_T = simulated[:, 3]

        for i in range(num_measured_data):
            measured_R  = measured[i, 1]
            measured_T  = measured[i, 2]

            #評価関数が最小になるのはどこ？
            nowg   = ((simulated_R - measured_R)/measured_R)**2 + ((simulated_T - measured_T) / measured_T)**2
            loc    = np.where(nowg==nowg.min())[0][0]
            result = np.r_[simulated[loc, 0], simulated[loc, 1], simulated[loc, 2], simulated[loc, 3]]

            #最小のときのエラーは？
            R_error = 100* np.abs(1-float(simulated_R[loc])/measured_R)
            T_error = 100* np.abs(1-float(simulated_T[loc])/measured_T)
            
            #エラーが0.5%以下とならない時、mua, musを新しいリストを作成
            if R_error > 0.5 or T_error > 0.5:
                judge = 1
                new_item = np.r_[simulated[loc, 0], simulated[loc, 1]].tolist()
                pair.append(new_item)
            else:
                judge = 0
            
            #結果を.xlsxに出力するためのリスト
            result = np.r_[measurement_data[i], result, R_error, T_error, judge].tolist()
            results.append(result)

        #終了条件
        if not pair:                  #mua, musのリストが空 = 全部0.5%以内
            task = 'Completed'
            return task, results
        else:
            task = 'Continue'
            return task, results, pair

    # 2
    def Create_mua_mus_forMC(self, rank):

        #ランク取得
        current_interval_mua = interval_mua['rank{:}'.format(rank)]           #noa
        current_interval_mus = interval_mus['rank{:}'.format(rank)]           #nos
        new_interval_mua     = interval_mua['rank{:}'.format(rank+1)]         #nea
        new_interval_mus     = interval_mus['rank{:}'.format(rank+1)]         #nes
        num_mua = int(current_interval_mua* 2 / new_interval_mua) + 1         #ja
        num_mus = int(current_interval_mus* 2 / new_interval_mus) + 1         #ka
        
        for i in range(len(data_pair)):
            #新しいintervalでmua, musのペアを決定（np.arangeは使わない方がいい）
            mua_next = np.linspace(data_pair[i][0] - current_interval_mua, data_pair[i][0] + current_interval_mua, num_mua)
            mus_next = np.linspace(data_pair[i][1] - current_interval_mus, data_pair[i][1] + current_interval_mus, num_mus)

            #0以下削除、データの桁数調整
            mua_next = np.array([round(mua, 9) for mua in mua_next if mua >= 0])
            mus_next = np.array([round(mus, 9) for mus in mus_next if mus >= 0])

            #mua=0だと回りにくいので
            np.putmask(mua_next, mua_next==0, 0.0001)

            #mua, musのセットの重複削除
            tmp_pair = np.array(list(itertools.product(mua_next, mus_next)))
            if i == 0:
                new_MC_pair = tmp_pair
            else:
                new_MC_pair = np.concatenate([new_MC_pair, tmp_pair])

            new_MC_pair_df = pd.DataFrame(new_MC_pair)
            new_MC_pair_df.drop_duplicates(inplace=True)
            new_MC_pair = new_MC_pair_df.to_numpy()
        
        return new_MC_pair

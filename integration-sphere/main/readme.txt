CUDAMCMLを使ったInverse Monte Carlo法による光学特性値解析の完全自動化ツールです。

Google ColaboratoryでGPUを使って回します。

Rd, Ttのcsvを指定すれば、ほぼワンクリックで自動的に計算が回ります。

Colab無料版を使用してる場合、Rd, Ttのcsvに何スペクトルもまとめないほうがいいです。
CUDA使用とはいえ、数時間かかる可能性があります。
長時間のGPUの使用は、GPUリソースの使い過ぎで強制切断されます。

光学特性値解析は1スペクトル（厚さやデータ数等にもよりますが）30分ほどで終了します。

SlackのURLを指定しておけば、計算が終わった段階で個人DMに通知が来ます。
個人によってURLは異なるので、各自のURLを指定してください。


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

各自のGoogleドライブには、フォルダ「配布用」からコピーしてください。

同一のRd, Ttスペクトルに対し、普通のMCMLとCUDAMCMLを使って光学特性値解析した比較データもあります。
「Example (MCML, CUDAMCML比較)」を見てください。

NASからコピーしたコードでエラーが出た場合は、高井まで教えてください。
コピーした後に各自でコードを変更した場合は対応できかねます。


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

【追記　2022/10/11】

2022年10月上旬にGoogle ColabのCUDAバージョンが変わったみたいです。

makefileの一部を変更してバージョンに合わせないとコンパイルできなくなります。

バージョンの変更などの理由で回らなくなり、修正に困った場合は、
yt.peter22@gmail.com（高井）まで連絡ください。
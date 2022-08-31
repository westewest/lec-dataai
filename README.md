[![「データシステムの知能化とデザイン」レクチャーノートブック](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai)
[![PyTorchバッジ](https://img.shields.io/badge/PyTorch->v1.0-232f3e.svg?style=flat)](https://pytorch.org/)
[![Slackデータシステムの知能化とデザイン](https://img.shields.io/badge/Slack-keio--sd--dataai-3f0f40.svg?style=flat)](https://keio-sd-dataai.slack.com/)
[![HIT-Academy](https://img.shields.io/badge/Slack-hitacademy--ml-3f0f40.svg?style=flat)](https://hitacademy-ml.slack.com/)

# 「データシステムの知能化とデザイン」レクチャーノートブック
## lec-dataai / Lecture Notes for "Design of Systems for Data and Artificial Intelligence"
- 全体の閲覧とダウンロード
  - 左上もしくはこちら[![「データシステムの知能化とデザイン」レクチャーノートブック](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai)のバッジを利用してください
- 個別ダウンロードは下記にリンクがあります
  - 個別ダウンロードはGitHubから直接Colaboratoryのサイトを開くことができるため、講義を受ける際にはこちらが便利です

### 授業はPCの利用（ブラウザの利用）が必須です
- 最新のChrome、Firefox、Edge、Safariなどが利用できる環境であれば問題ありません
- スマートフォンやタブレットでもPCと同等の機能を持つブラウザであれば利用できますが、作業効率などからお勧めしません
- 自分のパソコンを持参することを強くお勧めします

### Google Colaboratoryが提供する仮想マシン環境には様々な制限があります
- 実行時間などに制限があるため放置すると最初からやり直しになります
- Google Colaboratory(Colab)は再セットアップするたびにファイルが削除されます
- Google Driveをマウントすることもできますが、工夫しなければ基本的にはなにも保存されません

### Google Colaboratory(Colab)の利用について
- この授業は、Google Colaboratory (以下 Colab)を利用します
  - ColabはWebブラウザとGoogleアカウントがあれば利用できます
  - 内部でGoogle Driveを利用しますので、GoogleアカウントおよびDriveの利用設定が必要です
  - アカウントがあれば自宅でも、どこでも、同様に利用できます

## Colabの利用形態ついて
- Colabは無料で利用できる無償版の他、有償版があります
  - ここで扱う内容の確認、および課題・試験などに取り組む場合も無償版の利用で問題ありません
- 授業で扱う学習内容において、無償版と有償版の違いはありませんし、区別しません
  - 違うのは性能とインタフェースの一部だけで、基本機能は変わりません
- 有償版の方が素早く課題を終えることができる可能性があります
  - 有償版はより早いGPUを利用できる可能性が高いためです
  - 有償版は1000円と少し/月(2021年調査)で利用できるため、かなりお得で、十分に利用価値があります
- 講義や課題で利用する場合は日中の利用を推奨します
  - 海外、特にアメリカが利用する日本の夜間は混雑する傾向があります
  - 海外が夜間となる時間帯が空いています
  - 日中混雑して利用できなかったという報告を過去受けておらず、昨年も同様に試験が実施できました

## 個人環境の利用について
- 個人で専用のGPUマシンを揃えたうえで、Colab同様の環境を個人で構築して利用することができます
  - 能力次第でColabを利用せずに履修できますが、GPU環境の利用を必須とします
  - 構築そのものは自己責任で行って下さい
  - 授業としては構築サポートは行いません
  - 但し、構築に関する質問があれば、可能な範囲で対応します
- 互換性の問題による不具合・不利益は免責とします
  - 個人環境構築はいばらの道ですが、構築して利用した場合、最終評価で採点上考慮することがあります
  - 新たにGPU対応のPC、高価なGPUを購入するのは無意味で、Colabを使った方がよいです

## 授業テキスト
下記のColabのリンクをクリックすると、Colabを開くことができます
- 開いた後、必ず「ノートブックの保存」を行い、自身のGoogle Drive内部に保存してください

- 1-準備  
[![dataai-text-1-準備.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-1-準備.ipynb)

- 2-ML基礎  
[![dataai-text-2-ML基礎.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-2-ML基礎.ipynb)

- 2-ML基礎-補助  
[![dataai-text-2-ML基礎-補助.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-2-ML基礎-補助.ipynb)

- 2-python復習  
[![dataai-text-2-python復習.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-2-python復習.ipynb)

- 3-データの扱い  
[![dataai-text-3-データの扱い.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-3-データの扱い.ipynb)

- 4-MLライブラリの基礎  
[![dataai-text-4-MLライブラリの基礎.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-4-MLライブラリの基礎.ipynb)

- 5-Sktlearn-まとめ  
[![dataai-text-5-Sktlearn-まとめ.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-5-Sktlearn-まとめ.ipynb)

- 6-ニューラルネットワークの基礎  
[![dataai-text-6-ニューラルネットワークの基礎.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-6-ニューラルネットワークの基礎.ipynb)

- 7-PyTorch  
[![dataai-text-7-PyTorch.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-7-PyTorch.ipynb)

- 8-PyTorch-Basics  
[![dataai-text-8-PyTorch-Basics.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-8-PyTorch-Basics.ipynb)

- 9-PyTorch-CNN  
[![dataai-text-9-PyTorch-CNN.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-9-PyTorch-CNN.ipynb)

- A-PyTorch-RNN  
[![dataai-text-A-PyTorch-RNN.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-A-PyTorch-RNN.ipynb)

- B-PyTorch-AutoEncoder  
[![dataai-text-B-PyTorch-AutoEncoder.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-B-PyTorch-AutoEncoder.ipynb)

- C-PyTorch-転移学習  
[![dataai-text-C-PyTorch-転移学習.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-C-PyTorch-転移学習.ipynb)

- D-PyTorch-強化学習  
[![dataai-text-D-PyTorch-強化学習.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-D-PyTorch-強化学習.ipynb)

- E-PyTorch-説明可能AI  
[![dataai-text-E-PyTorch-説明可能AI.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-E-PyTorch-説明可能AI.ipynb)

- F-PyTorch-GAN-1  
[![dataai-text-F-PyTorch-GAN.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-F-PyTorch-GAN-1.ipynb)

- F-PyTorch-GAN-2  
[![dataai-text-F-PyTorch-GAN.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-F-PyTorch-GAN-2.ipynb)

- G-PyTorch-自然言語処理-1  
[![dataai-text-G-PyTorch-自然言語処理-1.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-G-PyTorch-自然言語処理-1.ipynb)

- G-PyTorch-自然言語処理-2  
[![dataai-text-G-PyTorch-自然言語処理-2.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-G-PyTorch-自然言語処理-2.ipynb)

- G-PyTorch-自然言語処理-3  
[![dataai-text-G-PyTorch-自然言語処理-4.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-G-PyTorch-自然言語処理-3.ipynb)
  
- G-PyTorch-自然言語処理-4  
[![dataai-text-G-PyTorch-自然言語処理-4.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-G-PyTorch-自然言語処理-4.ipynb)

- H-PyTorch-音声識別  
[![dataai-text-H-PyTorch-音声識別.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-H-PyTorch-音声識別.ipynb)

- I-PyTorch-StyleGAN-1  
[![dataai-text-I-StyleGAN-1.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-I-PyTorch-StyleGAN-1.ipynb)

- I-PyTorch-StyleGAN-2  
[![dataai-text-I-StyleGAN-2.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-I-PyTorch-StyleGAN-2.ipynb)

- I-PyTorch-StyleGAN-3  
[![dataai-text-I-StyleGAN-3.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-I-PyTorch-StyleGAN-3.ipynb)

- J-PyTorch-Diffusion  
[![dataai-text-I-StyleGAN-3.ipynb](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/keioNishi/lec-dataai/blob/main/dataai-text-J-PyTorch-Diffusion.ipynb)

## Cuda環境の構築について

Colabを利用すると様々な制約がありますが、自分のマシンに環境を構築することでこの制約を取り除くことができます
- ネットワークに繋がっていなくても実行できるようになります
- 高性能なGPUがあればColabよりもかなり実行時間を削減できます

自宅などのマシンに独自の環境を構築して利用するには、次を参考にチャレンジしてください
- 特に困らないであろうというところは、説明を省略しています。

- WindowsかUbuntu PCを準備
- Windowsの場合は以下の通りにインストール
  - WSL2 Ubuntu-20.04 LTSのインストール
Windowsマークを右クリック→Windowsターミナル（管理者）を立ち上げ、以下のコマンドラインからWSL2 Ubuntu-20.04 LTSをインストール 
> wsl --install -d Ubuntu-20.04 
  - NVIDIAドライバのインストール
NVIDIAのダウンロードサイトから、windows->x86_64->11->exe を選択してダウンロード 
  - コマンドラインに以下をいれて動作を確認
> nvidia-smi 
  - WSLの自動インストーラで一式導入するため、WSLのshellを起動 
> git clone https://github.com/tak6uch1/wsl2_pytorch_tf_gpu.git 
> cd wsl2_pytorch_tf_gpu 
> bash 1_install_cuda.sh 
  - CuDNNのインストール
NVIDIAのCuDNNダウンロードサイトをブラウザで開き、I Agree～にチェックを入れ、CUDA 11.5を選択 
 
Local Installer for Ubuntu20.04 x86_64[Deb] をダウンロード 
ダウンロードフォルダにcudnn-local-repo-ubuntu2004-8.3.2.44_1.0-1_amd64.debがダウンロードされる 

  - wsl2_pytorch_tf_gpuに移動して3_install_cudnn.shを実行
```
mv /mnt/c/Users/user_name/Downloads/cudnn-local-repo-ubuntu2004-8.3.2.44_1.0-1_amd64.deb .
bash 2_install_cudnn.sh
```
  - Anacondaのインストール
Anacondaのサイトからインストール用スクリプトをダウンロード 
Linux 64-Bit(x86) Installer を選択 
  - インストール用スクリプトを実行
bash /mnt/c/Users/user_name/Downloads/Anaconda3-2021.11-Linux-x86_64.sh 
user_nameは自身のアカウント名で置き換えること
  - Anacondaインストーラが~/.bashrcに設定を追記するため、sourceする
source ~/.bashrc
  - Tensorflow
Anacondaをインストールするとcondaコマンドが使えるようになる 
Tensorflow用の仮想環境をpython 3.8でtfとして作成し、アクティベート 
```
conda create -n tf python=3.8
conda activate tf
```
- Ubuntuへの導入
  - Cuda Toolkitをインストール

基本最新版で問題ありません。

  - cuDNNをインストール

こちらも問題ないはずです。アーキテクチャにより若干インストールの方法が違うようです。

  - Anacondaをインストール

これも問題ないはずです。

- 以降、Windows、Ubuntu共通で、まずAnaconda Navigatorを起動するなどして、新たな環境を構築

要するに、conda create するということです。

- Anaconda環境を更新する

```
conda config --set ssl_verify no
conda config --set allow_conda_downgrades true
conda update -y conda
conda install -y anaconda
conda update -y --all
```

- Pytorhをインストールする

toolkitバージョンはpytorthのホームページで確認してください
```
Linux
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia
もしくは
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
Windows
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c pytorch
もしくは
conda install -y pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
#(バージョンによってはこちらだがお勧めしない) conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
```
導入したら、次で動作を確認
```
python
>>> import torch
>>> print(torch.cuda.is_available())
True
```
最後にTrueと出ればOK、出ない場合は、頑張って解決しましょう  
例えば、間違ってcpu版が入っている可能性があります

- Jupyter Notebookをインストール

```
pip install jupyter
pip install --upgrade jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
jupyter nbextension enable --py widgetsnbextension
```

- 授業で利用するライブラリをインストール

```
conda install -y numpy
conda install -y pandas
conda install -y cudatoolkit
conda install -y cudnn
conda install -y keras
conda install -y scikit-learn scikit-learn-intelex
conda install -y scikit-image
conda install -y matplotlib
conda install -y python-graphviz
conda install -y pydotplus
conda install -y seaborn
conda install -y missingno
conda install -y lxml
```

- OpenCVをインストール

ちょっと大物なので、分けています  
conda install opencv はうまくいかない場合次のようにするとよいでしょう

```
pip install opencv-python
conda install -y -c conda-forge opencv
conda install -y -c conda-forge librosa
conda install -y -c conda-forge lightgbm
conda install -y -c conda-forge xgboost
conda install -y -c conda-forge ipywidgets
conda install -y -c conda-forge gensim
```

- そのほかの関連ライブラリをインストール

```
Linux
conda install -y -c conda-forge mecab
Windows
conda install -y -c mzh mecab-python3

pip install requests beautifulsoup4
```
- さらに次も自然言語処理で利用します
```
apt install libmecab-dev mecab mecab-ipadic-utf8
pip install mecab-python3
ln -s /etc/mecabrc /usr/local/etc/mecabrc
```

なお、次が必要となる場合もあります。
```
pip install --upgrade numpy
```

- tensorflowを入れる

これが、pythonのバージョン整合が厳しく、失敗することが比較的多いです。なお、Windows11 のWSL環境では、問題なくインストール可能です。
```
conda install -y tensorflow-gpu tensorflow-datasets tensorflow-hub
```
ですが入らなくても特に困ることはありません。
```
conda install -y tensorflow-gpu tensorflow-datasets tensorflow-hub -c conda-forge
```
で入る場合もあります。なお、tensorflow-gpuさえ入ればなんとかなります。

- quiltを入れる

これも、pythonのバージョンに厳しく、新しいpythonではインストールできないようです。といっても、利用するのは一か所だけなので、なくても大丈夫です。
```
conda install -y -c conda-forge quilt
quilt install --force ResidentMario/missingno_data
```

- おまけ

中にはwgetなど、Linux系のコマンドを利用しています  
Linux上で構築する場合は特に問題とはなりませんが、Windows上で構築するには、次の2つのLinuxで著名なコマンドラインツールを導入をしておくとよいでしょう

  - wgetの導入

使用頻度も高いので、ぜひ入れてください。

https://sourceforge.net/projects/gnuwin32/files/wget/1.11.4-1/wget-1.11.4-1-setup.exe/download

これを実行するだけです  
ほかのアーキテクチャでも同様に、利用できるようにしてください

  - Gitの導入
  
GitHub環境を自身のマシンに導入する際には、ほぼ必須ともいえるツールです  
特に、Windowsユーザの皆さんには、Git Bashの導入をお勧めします  
Git Bashを導入することで、下記、busyboxの導入は不要になるといえます

  - busyboxの導入

Git Bashを導入しない場合、Windowsでは、lsなどUnix系コマンドの実行はかなり厄介です(Windows11でかなり良くなりますが)  
そこで、次のbusyboxの導入が検討されますが、お勧めではありません(https://frippery.org/files/busybox/busybox.exe)

導入後、busybox.exeをC:\Windowsにコピーして、その中で busybox --installとするとメジャーコマンドが利用できるようになります

### Jupyter NotebookをGoogle Colaboratoryに接続する

これが重要です  
Google Colaboratoryに慣れている人は、ピュアなJupyter Notebookは使いにくいと感じると思います  
そこで、いつも通りGoogle Colaboratoryを利用しつつ、ローカルの計算リソースを利用することができますので、紹介します

- 最初だけ、実行バッチファイルを作成する

メモ帳でも、busyboxのviでもよいので、jupyterrun.batというファイルを作成します。中身は次の通りです
```
jupyter notebook --no-browser --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --NotebookApp.port_retries=0 --allow-root --ip=0.0.0.0 --NotebookApp.token=''
```
なお、この仕様は近々変更される予定で、
```
jupyter notebook --no-browser --ServerApp.allow_origin='https://colab.research.google.com' --port=8888 --ServerApp.port_retries=0 --allow-root --ip=0.0.0.0 --NotebookApp.token=''
```
とする必要があるかもしれません

- 最初に一度jupyterrun.batを実行する

ログが吐き出され動き出すはずです。これが動いている間は、複数のセッションが接続できます

- 普通にGoogle Colaboratoryでノートブックを開き、「接続」する際に、「ローカルランタイムに接続」を選択する

http://localhost:8888/ バックエンドに指定されているはずですので、そのまま接続とします

これで、 Google Coalbを利用せず、自分の環境を利用するようになります。全ての制限が外れます。つまり、利用時間やセッションの制限はなくなり、ファイルが消えることもありません

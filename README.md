# streamlit_demo

## Description
Streamlitのデモ共有

## Deploy to heroku
以下のサイトを参考にしてください。
* https://www.pluralsight.com/guides/deploying-image-classification-on-the-web-with-streamlit-and-heroku
* https://gilberttanner.com/blog/deploying-your-streamlit-dashboard-with-heroku
重要な点は、
1. Herokuサーバーで動くために必要なPythonライブラリをrequirements.txtで追加
2. Procfileとsetup.shを自分のherokuアカウント、アプリを動かすpythonファイル名に従って編集して追加
3. Heroku accountの作成とHeroku CLIのインストール
です。

* opencv-pythonをherokuで使うために[これ](https://qiita.com/haru1843/items/210cb08024195b9d1bc8)にしたがっていくつか設定変更する必要があります。
* Aptfile はopencv用の追加ファイル

## Localでの動かし方
1. このレポジトリのclone先に移動
2. （仮想環境で）requirements.txtにあるパッケージをpip等でインストール
3.  `streamlit run test.py`でLocalブラウザで実行

## Licence
[MIT](https://github.com/Taiki92777/streamlit_demo/blob/master/LICENSE)

## Author
[T.Harada](https://github.com/Taiki92777)

## Code reference
* [CheXNet](https://arxiv.org/abs/1711.05225)
* [Coursera AI for medicine](https://www.coursera.org/specializations/ai-for-medicine)
* [Streamlit documentation](https://docs.streamlit.io/en/stable/)

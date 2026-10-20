# 材料試料搬送ロボット — Physical AI応用1 最終課題の試作

密閉済みの模擬試料キャリアをカメラ画像で識別し、対応する分析装置の受付トレイへ搬送する。
CRANE+ V2の公式形状・関節・質量モデルをMuJoCoへ取り込み、画像識別CNNと搬送制御を接続した初期ベースライン。

## 確認済み環境

- Ubuntu 24.04 / Python 3.12、workspace内の独立した `.venv`。
- NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition、約96GB VRAM、compute capability 12.0。
- PyTorch 2.14.1+cu130。CUDA 13.0による行列演算とCNN学習を実行確認済み。
- MuJoCo 3.16.0。物理計算はCPU、ヘッドレス描画はOSMesaによるソフトウェア描画。
- 小さなCNNの実行時推論はCPU。GPUは学習に使用。
- ROS / Gazebo はこの試作環境にはインストールしていない。

## 現在の動作

1. 試料キャリア3本にA/B/Cの識別色を割り当てる。割当はseedでシャッフルする。
2. 上方カメラの実際のRGB画像を既知座標で切り出し、小型CNNでIDを分類。
3. CNNの出力から搬送先を決定する。配置位置だけを見た固定ルートでは解けない。
4. 逆運動学と関節位置制御で接近、把持、持ち上げ、搬送、解放、退避。
5. 全試料の最終配置、持ち上げ、関節制限、速度を確認し、JSONを保存。

色はキャリアのIDマーカー。画像から物質組成や危険性を推定するものではない。

## 再実行

構築済みの環境では以下で実行できる。

```bash
cd /workspace/physical_ai_sample_transport
# 学習済みCNNを使って、動画付きで搬送
scripts/run.sh scripts/demo.py --ai --video
# IDの配置を変えて評価
scripts/run.sh scripts/demo.py --ai --seed 7
# 4通りのID配置をまとめて評価
scripts/run.sh scripts/evaluate.py
# データ生成とGPU学習をやり直す
scripts/run.sh scripts/vision.py
# 元のxacroからモデルを再生成
scripts/run.sh scripts/build_scene.py
```

別のUbuntu 24.04 x86_64環境では、uv、Python 3.12、git、curl、dpkg-debが利用できる状態で `bash scripts/setup.sh` を実行する。OSライブラリは `.local` に展開し、システムや他プロジェクトの環境を変更しない。依存ライブラリのバージョンは `requirements-sim.txt` と `requirements.lock.txt` を参照。CUDA版PyTorchは約数GBを使用する。
動画の字幕にはDejaVu Sansを使う。元の生成済みassetsには絶対パスが含まれるため、移動後は `build_scene.py` で再生成する。

OSMesaのLLVMとPyTorch/Tritonのライブラリが同一プロセスに読み込まれると衝突する環境のため、画像生成とGPU学習を別プロセスに分けている。学習プロセスでは `MUJOCO_GL=disable` を設定する処理を実装済み。学習には通常 `scripts/run.sh scripts/vision.py` を使う。

## 初回の確認結果

IDの割当を変えた4回・計12本の搬送で12本成功。関節制限の超過なし。試料の位置は固定で、把持も簡易拘束のため、未知配置や実機での成功率を意味しない。画像識別は独立した12シーン由来の288枚で100%（模擬画像のみ）。動画はH.264 / 1280×720 / 15fps、63.07秒。

## ファイル

- `scripts/build_scene.py`: 公式xacro展開、MuJoCo変換、試料・トレイ・カメラを追加。
- `scripts/control.py`: IK、関節制御、近接確認付き簡易把持。
- `scripts/vision.py`: シミュレーション画像生成、CNN学習・推論。
- `scripts/demo.py`: 画像識別から搬送までの実行、動画・評価JSON。
- `artifacts/baseline_demo.mp4`: 初期動作確認動画。提出用の最終動画ではない。
- `artifacts/result_seed_*.json`: 各実行の搬送結果。
- `artifacts/vision_metrics.json`: 模擬画像の識別評価。
- `artifacts/marker_cnn.pt`: 学習済み重み。
- `docs/project_plan.md`: テーマ・評価方法・次の開発項目。

## 評価方法と制約

学習用は30シーン、評価用は独立したseedの12シーン。各シーンの3本について8通りの露光・ノイズ変動を作る。720枚で学習、288枚で評価する。シーンごとに識別色・照明・数mmの位置を変える。評価用画像も模擬画像であり、同一シーン由来の8枚は相関がある。実試料や実環境への汎化は未評価。

搬送成功は正しいIDのトレイへ配置、XY誤差15mm未満、中心高さ24mm±9mm、持ち上げ中心高さ45mm超、配置直後の静止を条件とする。動作終了後に全試料の配置を再確認する。関節角制限の超過1mrad未満・最大速度2rad/s未満も確認する。

**把持は初期実装として簡易化している。** 先端と試料中心の距離8mm以内を確認してから現在の相対姿勢でweld拘束を有効化する。物体の座標を手先へ直接書き換える処理はない。試料と机・トレイ・他試料の接触と重力は有効。ロボットと試料の接触、およびロボットの自己接触は無効。ロボットと机・トレイの接触は有効。摩擦だけで成立する把持を検証した結果として扱わない。

試料位置はシミュレータから取得する。CNNが担うのは画像中のID識別であり、位置推定・把持計画・制御全体を学習したモデルではない。搬送中は可動域に合わせ容器が最大約37度傾く。対象は密閉済みキャリアとし、開放容器や液面制御は対象外。重力補償付き位置制御を使用するため、実機のモータ制御をそのまま再現したものでもない。

## 参照・帰属

- 課題の着想元: https://index.ros.org/p/crane_plus_gazebo/
- CRANE+ V2モデル: https://github.com/rt-net/crane_plus
- 取得revision: `34c4584c7373463f319174bb76a2198943ab9cb8`。
- CRANE+パッケージはApache-2.0。元ファイルのライセンスと、ROBOTIS製部品CADに関する権利表記を保持する。詳細は `THIRD_PARTY_NOTICES.md` および `vendor/crane_plus/LICENSE`。
- MuJoCo: https://mujoco.readthedocs.io/

本試作の独自部分は試料シーン、ID認識CNN、画像から搬送先を選ぶ処理、IKと搬送手順、検証・動画生成処理。公式Gazeboのサンプル実行をそのまま提出する構成ではない。

# 一口馬主 AI予測システム (JRA Racehorse Predictor Pro)

> *English version is available in the bottom half of this document.*

JRA（日本中央競馬会）の過去の競走馬データと血統情報を学習した機械学習モデル（LightGBM）を用いて、競走馬が将来到達する可能性の高い「適性クラス（未勝利、1勝クラス〜オープン等）」を予測するWebアプリケーションです。一口馬主の出資馬選びや、競馬のデータ分析・研究のサポートを目的として開発されています。

## 🌟 主な機能

- **適性クラス予測**: 性別、誕生月、厩舎、馬主、生産者、種牡馬などのデータを入力することで、AIが各クラスへの到達確率を算出・可視化します。
- **直感的なUI**: Glassmorphismデザインを取り入れたモダンな画面設計。アニメーション付きのプログレスバーで確率を一目で把握できます。
- **サジェスト機能**: 厩舎や騎手、馬主などの入力フィールドは、学習済みデータからの予測変換（オートコンプリート）に対応しており、入力の手間を省きます。
- **予測ログの自動保存**: ユーザーが実行した予測データはFirebase (Firestore) に自動的に保存され、後からの検証や機能改善に役立てることが可能です。

## 🛠️ 技術スタック

- **バックエンド**: Python 3.10, FastAPI, Uvicorn
- **機械学習**: LightGBM, scikit-learn, Pandas, Joblib
- **フロントエンド**: HTML5, Vanilla CSS, Vanilla JavaScript
- **データベース（ログ保存）**: Firebase Firestore (Google Cloud)
- **インフラ・デプロイ**: Docker, Google Cloud Run
- **テスト**: Pytest, HTTPX

## 🏗️ システムアーキテクチャ・設計上の工夫

- **推論の高速化とUX**: 軽量なLightGBMモデルを採用し、バックエンド（FastAPI）でミリ秒単位の推論を実現。さらにフロントエンド側では非同期的（非ブロッキング）にAPIを呼び出すことで、画面のフリーズを防ぎ、ローディングアニメーションと同時に結果を返すスムーズなUXを提供しています。
- **スケーラブルなインフラ構成**: Google Cloud Run（サーバーレスコンテナ）を採用。ゼロスケールによるコスト最適化と、高負荷時の自動スケールアウトによる可用性を両立させています。
- **データ駆動型のサジェスト機能**: ユーザーの入力負荷を軽減するため、事前に加工済み（前処理済み）の特徴量データセット（CSV）から一意な候補リストを生成し、フロントエンドにキャッシュさせることで高速な予測変換（オートコンプリート）を実現しています。
- **非同期データロギング**: 分析の精度向上とユーザー動向の把握のため、推論結果と入力データをFirebase Firestoreへ記録しています。この際、FastAPIの `BackgroundTasks` を利用し、推論のレスポンス完了後に裏側でDBへの通信（非同期保存）を行うことで、ユーザーへの結果表示を一切遅延させない設計としています。

## 🔒 セキュリティと品質保証

- **自動テストの導入**: `pytest` および `httpx` を用いた結合テスト（APIテスト）を整備し、バリデーションエラー時の適切なステータスコード（HTTP 422など）の返却や、GET/POSTメソッドの保護を確認しています。

## ⚠️ 免責事項
当アプリケーションおよび付随する機械学習モデル・コードは、個人の研究・学習ならびに技術ポートフォリオを目的として開発されたものです。予測結果は該当馬の将来の競走成績・獲得賞金・適性等をいかなる意味においても保証するものではありません。

---

# JRA Racehorse Predictor Pro

This is a web application that uses a machine learning model (LightGBM) trained on historical horse racing data and pedigree information from the Japan Racing Association (JRA) to predict the most likely "aptitude class" (e.g., Maiden, 1-Win Class ... Open Class) a racehorse will reach in its career. It is developed to support racehorse syndicate investments (I一口馬主/Hitokuchi Banushi), as well as general horse racing data analysis and research.

## 🌟 Key Features

- **Aptitude Class Prediction**: By inputting data such as gender, birth month, stable, owner, breeder, and sire, the AI calculates and visualizes the probability of the horse reaching each class.
- **Intuitive UI**: A modern screen design incorporating Glassmorphism. Users can instantly grasp probabilities through animated progress bars.
- **Auto-Suggest Functionality**: Input fields for stables, owners, breeders, etc., support predictive text (autocomplete) based on the training data, saving input time.
- **Automated Prediction Logging**: The prediction data executed by the user is automatically saved to Firebase (Firestore), which can be used for later verification and feature improvement.

## 🛠️ Tech Stack

- **Backend**: Python 3.10, FastAPI, Uvicorn
- **Machine Learning**: LightGBM, scikit-learn, Pandas, Joblib
- **Frontend**: HTML5, Vanilla CSS, Vanilla JavaScript
- **Database (Logging)**: Firebase Firestore (Google Cloud)
- **Infrastructure & Deployment**: Docker, Google Cloud Run
- **Testing**: Pytest, HTTPX

## 🏗️ System Architecture & Design Highlights

- **Inference Optimization & UX**: Utilizes a lightweight LightGBM model to achieve millisecond inference times on the backend (FastAPI). On the frontend, the API is called asynchronously (non-blocking) to prevent screen freezing, providing a smooth UX that returns results simultaneously with the loading animation.
- **Scalable Infrastructure**: Deployed on Google Cloud Run (serverless containers). This architecture balances cost optimization through scale-to-zero and high availability through automatic scale-out during heavy loads.
- **Data-Driven Auto-Suggest**: To reduce user input burden, unique candidate lists are generated from pre-processed feature datasets (CSV) and cached on the frontend, realizing high-speed autocomplete.
- **Asynchronous Data Logging**: To improve analysis accuracy and understand user trends, inference results and input data are recorded in Firebase Firestore. The FastAPI `BackgroundTasks` feature is used to communicate with the DB (asynchronous saving) in the background after returning the inference response, ensuring zero delay in displaying results to the user.

## 🔒 Security & Quality Assurance

- **Automated Testing Suite**: Implemented integration tests (API tests) using `pytest` and `httpx` to confirm the return of appropriate status codes (e.g., HTTP 422) during validation errors and the protection of GET/POST methods.

## ⚠️ Disclaimer
This application and its accompanying machine learning models and code were developed for the purpose of individual research, learning, and as a technical portfolio. The prediction results do not in any way guarantee the future racing performance, prize money earned, or aptitude of the horse in question.

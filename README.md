# 🧠 MBTI Personality Predictor (MBTI 文本性格預測系統)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Flask](https://img.shields.io/badge/Flask-Web%2B-green)

> **一句話簡介**：本專案基於 **BERT 預訓練語言模型** 與 **自然語言處理 (NLP)** 技術，透過分析社群平台上的使用者留言文本，自動預測並分類使用者的 MBTI 十六型人格。

<img src="https://github.com/yunjiee/MBTI_project/blob/main/static/pictures/1.jpg" width="50%">

## 📖 專案目標與說明

本文探討了如何使用自然語言模型，基於使用者的發文與標記文本，預測其 MBTI 性格類型。
我們從網路論壇爬取了大量 16 型人格的真實留言作為訓練數據，並建立了一個端到端 (End-to-End) 的應用程式——從**自動化爬蟲、資料前處理、BERT 模型微調訓練，一直到部署成 Flask 互動式 Web 網站**。

## ✨ 核心技術與功能

1. **🕸️ 資料蒐集與網路爬蟲 (Web Scraping)**
   * 使用 Python 的 `Selenium` 與 `BeautifulSoup` 套件。
   * 從外國知名論壇（如 PersonalityCafe）精確抓取 16 種人格的大量留言，建立高質量的自定義數據集（總計超過 6400 筆資料）。
2. **🧹 資料清洗與前處理 (Data Preprocessing)**
   * 透過正則表達式 (Regex) 與標準化技術，清除網址、雜訊、多餘空白及標點符號。
   * 將特定字眼替換為 `<type>` 以避免模型作弊，並過濾過短的無效文本。
   * 使用 `matplotlib` 與 `pandas` 進行 EDA（探索性資料分析）與數據分布視覺化。
3. **🤖 深度學習與 BERT 微調 (Model Fine-Tuning)**
   * 基於 `PyTorch` 與 HuggingFace `Transformers` 的 `BertForSequenceClassification`。
   * 憑藉 BERT 強大的上下文解讀能力，對留言進行語義分析與隱含情感捕捉。
   * 支援 Google Colab 環境進行 GPU 加速訓練 (`run.ipynb` / `fine_tune_save_colab.py`)。
   * 同時也實作了基於 TF-IDF 與 XGBoost (`xgboost_bert.py`, `xgboost_text.py`) 的對照實驗模型。
4. **🌐 互動式 Web 應用程式 (Flask Web App)**
   * 使用 `Flask` 框架建立直覺的 Web 介面 (`app.py`)。
   * 使用者只需在網頁輸入一段文字，後端即會呼叫訓練好的 BERT 模型進行即時推論，並返回詳細的 MBTI 性格分析結果。

## 📂 專案目錄結構

```text
MBTI_project/
├── app.py                  # Flask Web 應用程式主程式 (啟動網站)
├── out.py                  # 負責載入訓練好的 BERT 模型並進行預測推論
├── clean.py                # 文本前處理邏輯 (正則表達式清理雜訊)
├── 16personality_urls.txt  # 爬蟲目標網址清單
├── scraper/                # 網路爬蟲模組
│   ├── try_personality_crawler.py # 爬蟲主邏輯測試腳本
│   ├── infp_scraper.py, ...       # 16 型人格各自的爬蟲腳本
├── full/                   # 模型訓練與實驗核心代碼
│   ├── data/               # 存放訓練集 (train.csv)、驗證集 (dev.csv)
│   ├── bert_model.py       # 定義 BERT 模型架構與優化器 (AdamW)
│   ├── dataloader.py       # PyTorch DataLoader 數據批次處理
│   ├── processor.py        # 資料格式轉換 (轉為模型輸入格式)
│   ├── fine_tune.py        # 模型訓練主程式
│   ├── xgboost_*.py        # XGBoost 與 TF-IDF 的實驗對照代碼
│   └── eda.py              # 資料視覺化與分佈探索
├── templates/              # Web 前端 HTML 模板 (首頁、結果頁、說明頁)
└── static/                 # Web 前端靜態資源 (CSS 樣式、JS 腳本、圖片)
```

## 🚀 如何運行 (Getting Started)

### 1. 安裝環境與相依套件
請確保已安裝 Python 3.8+，並安裝以下必要套件：
```bash
pip install flask pandas numpy torch torchvision transformers xgboost scikit-learn beautifulsoup4 selenium nltk
```

### 2. 啟動 Web 應用程式
在專案根目錄下執行以下指令以啟動 Flask 伺服器：
```bash
python app.py
```
接著在瀏覽器打開 `http://127.0.0.1:5000` 即可體驗 MBTI 文字預測！

### 3. 模型訓練 (進階)
若要重新訓練模型，可進入 `full/` 目錄並執行微調腳本（建議使用 GPU 環境如 Google Colab）：
```bash
python full/fine_tune.py --do_train --do_eval
```

## 📊 結果與未來展望

**當前模型成效：**
* 準確預測所有 4 個性格維度（Exact Match）準確率達到 **0.47**。
* 正確預測至少 2 個性格維度準確率達到 **0.86**。
* 展示了微調 BERT 模型在性格特定語言生成中的高效應用，對現代心理學和智能共情系統具重要意義。

**待精進方向：**
* 📈 **指標擴充**：引入 Accuracy、AUC、Confusion Matrix (標準化混淆矩陣) 等更全面的指標來交叉驗證結果。
* 🧠 **向量空間分析**：將「文章向量」轉化為數字表示，探索其在運算、比較與聚類上的潛力。
* 🌲 **混合模型**：進一步結合 Logistic Regression 或 XGBoost 增強分類邊界。

## 📚 參考資料

1. Kaggle MBTI Datasets
2. *Myers-Briggs Personality Classification and Personality-Specific Language Generation Using Pre-trained Language Models*


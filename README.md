MBTI 性格指標  
  
目標:  
本文探討了使用自然語言模型來基於標記文本，預測在社群留言中的人的 MBTI 性格類型。  
  
使用技能:  
1. 資料蒐集與網路爬蟲：使用 Python 中的 Selenium 套件進行網路爬蟲，精確地抓取來自社群平台的大量留言數據，建立高質量的數據集。  
2. 資料清洗與前處理：通過資料清洗和標準化技術，對留言進行預處理，清除雜訊並對數據進行規範化處理。隨後，使用視覺化工具檢查數據分布，確保數據質量。  
3. BERT 模型應用：憑藉 BERT 模型的上下文解讀能力，對留言進行精確的語義分析，通過預訓練和微調模型，有效捕捉文本中的語義結構和隱含情感。  
4. MBTI 分類模型：模型依序對 MBTI 的四個核心維度（外向/內向、感覺/直覺、思維/情感、判斷/知覺）進行二分類判斷，最終確定個體在 16 種人格類型中的歸屬。這一過程顯示了模型在自然語言處理中的高效應用。  
  
結果:  
所提模型在正確預測所有四個類型方面達到 0.47 的準確率，在正確預測至少兩個類型方面達到 0.86 的準確率。此外，我們還研究了微調的 BERT 模型在性格特定語言生成中的可能應用。這是現代心理學和智能共情系統的重要任務。  

待精進方向  
=> 可以再透過Accuracy、AUC、Confusion Matrix(標準化混淆矩陣)或是Logistic Regression 去驗證結果  
=>「文章向量」就是把一篇文章轉成一串數字（向量），讓電腦可以運算、比較、分類。  

<img src="https://github.com/yunjiee/MBTI_project/blob/main/static/pictures/1.jpg" width="50%">



參考資料  
1.kaggle
2.Myers-Briggs Personality Classification and Personality-Specific Language Generation Using Pre-trained Language Models

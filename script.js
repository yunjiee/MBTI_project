function checkAndSubmit() {
    // 取得文章內容
    var articleContent = document.getElementById("articleInput").value;
  
    // 檢查字數是否達到50字以上
    if (articleContent.length >= 50) {
      // 將文章內容放入名為user_post的變數
      var user_post = articleContent;
      console.log("文章已提交: " + user_post);
      
      // 清空錯誤訊息
      document.getElementById("errorMessage").textContent = "";
    } else {
      // 顯示錯誤訊息
      document.getElementById("errorMessage").textContent = "文章內容需輸入50字(含)以上";
    }
  }
  
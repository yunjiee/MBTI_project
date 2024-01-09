function navigateToTheme(theme) {
    // 實現主題切換的相應操作
    console.log('切換到主題:', theme);
}

function checkAndSubmit() {
    var articleInput = document.getElementById("articleInput");
    var errorMessage = document.getElementById("errorMessage");
    var completionMessage = document.getElementById("completionMessage");

    // 取得文章內容
    var articleContent = articleInput.value;

    // 檢查字數是否達到50字以上
    if (articleContent.length >= 50) {
        // 將文章內容放入名為message的變數
        var message = articleContent;
        console.log("文章已提交: " + message);

        // 清空錯誤訊息
        errorMessage.textContent = "";

        // 清空完成訊息
        completionMessage.textContent = "";

        // 跳轉到等待頁面
        window.location.href = "waiting.html";
    } else {
        // 顯示錯誤訊息
        errorMessage.textContent = "文章內容需輸入50字(含)以上";
    }
}

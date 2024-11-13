const chatWindow = document.getElementById("chat-window");
const chatHistory = document.getElementById("chat-history");
const userInput = document.getElementById("user-input");

function addMessageToChat(content, isUser = false) {
    const messageElement = document.createElement("div");
    messageElement.className = isUser ? "message user-message" : "message bot-message";
    messageElement.textContent = content;
    chatHistory.appendChild(messageElement);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function sendMessage() {
    const message = userInput.value;
    if (message.trim() === "") return;
    addMessageToChat(message, true);
    userInput.value = "";

    fetch("/get_response", {
        method: "POST",
        body: new URLSearchParams({ message: message })
    })
    .then(response => response.json())
    .then(data => addMessageToChat(data.response));
}

function sendImage() {
    const fileInput = document.getElementById("file-upload");
    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("/upload_image", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => addMessageToChat(data.response));

    fileInput.value = "";
}

function closeChat() {
    document.querySelector(".chat-container").style.display = "none";
}

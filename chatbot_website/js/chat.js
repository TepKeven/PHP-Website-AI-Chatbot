function toggleChatPopup(event) {
  var chatbox_popup = document.getElementById("chatbox-popup");
  chatbox_popup.classList.toggle("d-none");

  event.target.classList.add("d-none");
}

function closeChatPopup(event) {
  var chatbox_popup = document.getElementById("chatbox-popup");
  chatbox_popup.classList.add("d-none");

  chatbox_popup.previousElementSibling.classList.remove("d-none");
}

function createChatBubble(row_class, text_class, text_content){

    var content = document.getElementById("content");
    var chat_row = document.createElement("div")
    chat_row.classList = row_class
    var text = document.createElement("p")
    text.textContent = text_content
    text.setAttribute("class", text_class)
    chat_row.appendChild(text);
    content.appendChild(chat_row)    
    content.scrollTop = content.scrollHeight;
}

function sendChat() {

  // User Chat
  createChatBubble("d-flex justify-content-end", "user-chat", document.getElementById("txt_chatbox").value)

  $.ajax({
    url: "http://localhost:8001/chat/",
    type: "POST", 
    data: { 
        chat: document.getElementById("txt_chatbox").value
    }, 
    success: function (response, status, xhr) {

        // Bot Chat
        createChatBubble("d-flex justify-content-start", "bot-chat", response.chat)
    },
    error: function (jqXhr, textStatus, errorMessage) {
        console.log(errorMessage);
    },
  });
}

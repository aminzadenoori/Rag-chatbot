css = '''
<style>
.chat-container {
    background-image: url('/Users/amin/ask-multiple-pdfs/background.png');
    background-size: cover;
    background-position: center;
    padding: 2rem;
    border-radius: 0.5rem;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
}

.chat-message.user {
    background-color: #2b313e;
}

.chat-message.bot {
    background-color: #475063;
}

.chat-message .avatar {
    width: 20%;
}

.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.cambridge.org/elt/blog/wp-content/uploads/2020/08/GettyImages-1221348467-e1597069527719.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://sp-ao.shortpixel.ai/client/q_glossy,ret_img,w_1940/https://s22908.pcdn.co/wp-content/uploads/2022/02/what-are-bots.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

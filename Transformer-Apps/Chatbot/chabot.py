from transformer import TranformerChatbot

chatbot = TransformerChatbot("microsoft/DialoGPT-medium")
while True:
    prompt = input("You: ")
    response = chatbot.generate_text(prompt)
    print("Chatbot:", response)

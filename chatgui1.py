import random
import json
import torch
import tkinter as tk
from tkinter import scrolledtext, END

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jbot"


def get_response(input_text):
    sentence = tokenize(input_text)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."


def send_message():
    message = user_input.get("1.0", END).strip()
    user_input.delete("1.0", END)

    if message != "":
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "You: " + message + "\n")
        chat_history.config(state=tk.DISABLED)

        bot_response = get_response(message)

        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, bot_name + ": " + bot_response + "\n")
        chat_history.config(state=tk.DISABLED)
        chat_history.yview(tk.END)


# Create the main window
window = tk.Tk()
window.title("Chatbot")

# Create a text area for displaying the chat history
chat_history = scrolledtext.ScrolledText(window, width=50, height=20, state=tk.DISABLED)
chat_history.pack(pady=10)

# Create a frame to hold the input text box and send button
input_frame = tk.Frame(window)
input_frame.pack()

# Create an input text box for the user with a hint
user_input = tk.Text(input_frame, width=50, height=3)
user_input.pack(side=tk.LEFT, padx=5)
user_input.insert(tk.END, "Type your message here...")

# Create a button to send the user's message
send_button = tk.Button(input_frame, text="Send", command=send_message)
send_button.pack(side=tk.LEFT, padx=5)

# Start the main event loop
window.mainloop()
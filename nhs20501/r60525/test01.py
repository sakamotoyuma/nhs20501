def test01():
    import nltk
    from nltk.chat.util import Chat, reflections

    pairs = [
        [r"hi|hello", ["Hello! How can I help you?"]],
        [r"(.*) your name?", ["I am a chatbot created using Python."]],
        [r"how are you (.*)", ["I'm a bot, I'm always functioning as expected"]],
        [r"quit", ["Goodbye!"]],
    ]

    def chatbot():
        print("Hi, I'm the chatbot!")
        chat = Chat(pairs, reflections)
        chat.converse()

    if __name__ == "__main__":
        chatbot()
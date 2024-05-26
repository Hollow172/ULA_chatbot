import functions

# chatbot using get_response_extractOne
def chat_bot1(file_path):
    """
    Implements a chatbot using process.extractOne for response retrieval.

    Parameters:
    - file_path (str): The path to the JSON file containing questions and answers.
    """
    data = functions.load_json(file_path)
    questions = data['questions']
    score_threshold = 85
    print("Chatbot: Welcome to chatbot program. Press 'quit' to quit the program.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Getting the response
        response, score = functions.get_response_extractOne(inp, questions)

        if score >= score_threshold:
            print("ChatBot: ", response)
        else:
            print("ChatBot: I don't understand you. Can you give me an explanation? (Type 'skip' to skip explanation)")
            ans = input("You: ")
            if ans.lower() == 'skip':
                continue
            data["questions"].append({"question": inp, "answer": ans})
            functions.save_json(file_path, data)
            print("ChatBot: Thank you for your explanation!")


# chatbot using get_response_cosine
def chat_bot2(file_path):
    """
    Implements a chatbot using cosine similarity for response retrieval.

    Parameters:
    - file_path (str): The path to the JSON file containing questions and answers.
    """
    data = functions.load_json(file_path)
    questions = data['questions']
    score_threshold = 0.35
    print("Chatbot: Welcome to chatbot program. Press 'quit' to quit the program.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Getting the response
        response, score = functions.get_response_cosine(inp,questions)

        if score >= score_threshold:
            print("ChatBot: ", response)
        else:
            print("ChatBot: I don't understand you. Can you give me an explanation? (Type 'skip' to skip explanation)")
            ans = input("You: ")
            if ans.lower() == 'skip':
                continue
            data["questions"].append({"question": inp, "answer": ans})
            functions.save_json(file_path, data)
            print("ChatBot: Thank you for your explanation!")


# chatbot using get_response_close_matches
def chat_bot3(file_path):
    """
    Implements a chatbot using close matches for response retrieval.

    Parameters:
    - file_path (str): The path to the JSON file containing questions and answers.
    """
    data = functions.load_json(file_path)
    questions = data['questions']
    score_threshold = 0.35
    print("Chatbot: Welcome to chatbot program. Press 'quit' to quit the program.")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        # Getting the response
        response, score = functions.get_response_close_matches(inp,questions, score_threshold)

        if score >= score_threshold:
            print("ChatBot: ", response)
        else:
            print("ChatBot: I don't understand you. Can you give me an explanation? (Type 'skip' to skip explanation)")
            ans = input("You: ")
            if ans.lower() == 'skip':
                continue
            data["questions"].append({"question": inp, "answer": ans})
            functions.save_json(file_path, data)
            print("ChatBot: Thank you for your explanation!")

# main part of the program - main menu
if __name__ == "__main__":
    while True:
        print("==================================")
        print("Available versions of the chatbot:")
        print("1. Chatbot using process.extractOne")
        print("2. Chatbot using cosine similarity")
        print("3. Chatbot using get_close_matches")
        choice = input("Enter 1, 2, or 3 to choose version of the chatbot(type 'quit' to exit): ")
        print("")

        if choice == '1':
            chat_bot1("questions.json")
        elif choice == '2':
            chat_bot2("questions.json")
        elif choice == '3':
            chat_bot3("questions.json")
        elif choice.lower() == 'quit':
            print("Exiting the program.")
            break
        else:
            print("There is no such option. Please try again.")
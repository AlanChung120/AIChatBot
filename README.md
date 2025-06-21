# AIChatBot

Finished and uploaded June 21, 2025. Used Python 3.12.5. This is a simple AI chatbot that acts as a friend. It trains the chatbot using intents.json (you can adjust to fit your use). For each training data phrase, it tokenizes, stems, and converts to a bag-of-words vector. Then it trains the model (simple neural network with two layers, which we can change) to classify the bag-of-words vector into a contextual tag (ex. greetings, compliments). Finally user can chat with the trained chatbot that responds based on the classified contextual tag determined by the output of the model when you input the user's typed input into the model.  
To Train:  
Run the train.py file using installed Python.  
To Run:  
Run the chat.py file using installed Python.

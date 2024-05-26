# ULA_chatbot
Project for Bachelor's Diploma Thesis - Chatbot "ULA" User Learning Assistant

# Instruction how to run chatbot on windows
# Step 1: Create a virtual environment named 'venv' in the current directory
python -m venv venv

# Step 2: Change directory to the 'scripts' folder within the 'venv' directory
cd venv/scripts

# Step 3: Activate the virtual environment
activate

# Step 4: Go back to the main project directory (two levels up from the 'scripts' directory)
cd ..
cd ..

# Step 5: Install all required dependencies listed in the 'requirements.txt' file
pip install -r requirements.txt

# Step 6: Run the 'additional_downloads.py' script to perform additioanl required setup
python additional_downloads.py

# Step 7: Run the 'chatbot.py' script to start the program
python chatbot.py

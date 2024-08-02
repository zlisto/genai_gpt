import os
from genai import GenAI  # Import the GenAI class correctly

# Import local API keys from config.py, uncomment if not deploying to Heroku
from config import *  

if __name__ == "__main__":
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

    jarvis = GenAI(OPENAI_API_KEY)

    instructions = "You are Jarvis, my AI assistant"
    prompt = "What is your name and purpose?"

    response = jarvis.generate_text(prompt, instructions)  # Corrected method call

    print(f"\nJarvis: {response}")


    filepath_paper = 'data/NIPS-2017-attention-is-all-you-need-Paper.pdf'

    text = jarvis.read_pdf(filepath_paper)

    instructions = "You are Jarvis, my AI assistant"
    prompt = f"What is the title of this paper?\n{text}"

    #response = jarvis.generate_text(prompt, instructions)  # Corrected method call
    #print(f"Jarvis: {response}")
    
    filepath_image = 'data/kendric_la_popup.jpg'
    instructions = '''What would a Kendrick Lamar seeing 
        this image live in person in the front row shout?'''

    #response = jarvis.generate_image_description(filepath_image, instructions)
    #print(f"\nJarvis: {response}")

    filepath_video = 'data/superbon_teeping.mp4'
    instructions = '''Describe whats happening in this video'''

    response = jarvis.generate_video_description(filepath_video, instructions)
    print(f"\nJarvis: {response}")

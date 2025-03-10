Nour Helmy - 202202012
## CSAI 422: Lab Assignment 3 - Taming LLMs with Groq API

The goal of this assignment is to build a content classification and analysis tool using the Groq API. The tool classifies text into categories, extracts key insights, and compares different prompt strategies.

## Features
Content Classification:
	- Classifies text into predefined categories (e.g., Positive, Negative, Mixed).
	- Analyzes the model's confidence in its predictions.

  Structured Completions:
	- Extracts specific sections from model completions using recognizable patterns.

  Prompt Strategy Comparison:
	- Compares different prompt strategies (basic, structured, few-shot) for classification tasks.

  Bonus Challenges:
	- Calibration function for confidence thresholds.
	- Comparison of results across different Groq models.
	- Streamlit Web Interface:
		- Provides an interactive web interface for text classification.

## Setup Instructions
  1. Prerequisites
  	- Python 3.8 or higher.
  	- A Groq API key (sign up at Groq's website).
  
  2. Install Dependencies
  	- Run the following command to install the required Python libraries:
  		pip install groq python-dotenv streamlit
  
  3. Set Up the .env File
  	- Create a .env file in the root directory of the project.
  	- Add your Groq API key to the .env file:
  	- GROQ_API_KEY=your_api_key_here

## Running the Script
  - To run the script and interact with the Groq API, execute the following command:
		  python taming_llm.py

## Running the Streamlit App
  - To launch the Streamlit web interface, run:
      streamlit run taming_llm.py
  - Open your browser and navigate to the URL provided in the terminal (usually http://localhost:8501).
  - Enter text in the input box and click the "Classify" button to see the results.

## Code Structure
  - Main Script: taming_llm.py
  - LLMClient Class: Handles interactions with the Groq API.
  - classify_with_confidence Function: Classifies text and analyzes model confidence.
  - compare_prompt_strategies Function: Compares different prompt strategies.
  Bonus Challenges:
    - calibrate_threshold: Tunes confidence thresholds based on test data.
    - compare_models: Compares results across different Groq models.
    - run_streamlit_app Function: Launches the Streamlit web interface.



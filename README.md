# Taming LLMs with Groq API

The goal of this assignment is to build a content classification and analysis tool using the Groq API. The tool classifies text into categories, extracts key insights, and compares different prompt strategies.

## Features
Content Classification:<br>
- Classifies text into predefined categories (e.g., Positive, Negative, Mixed).<br>
- Analyzes the model's confidence in its predictions.

Structured Completions:<br>
- Extracts specific sections from model completions using recognizable patterns.

Prompt Strategy Comparison:<br>
  - Compares different prompt strategies (basic, structured, few-shot) for classification tasks.

Bonus Challenges:<br>
  - Calibration function for confidence thresholds.<br>
  - Comparison of results across different Groq models.<br>
  - Streamlit Web Interface:<br>
  - Provides an interactive web interface for text classification.<br>

## Setup Instructions
  1. Prerequisites<br>
		- Python 3.8 or higher.<br>
  		- A Groq API key (sign up at Groq's website).<br>
  
  2. Install Dependencies<br>
  		- Run the following command to install the required Python libraries:<br>
  		   #### ***pip install groq python-dotenv streamlit***
  
  3. Set Up the .env File<br>
	  - Create a .env file in the root directory of the project.<br>
	  - Add your Groq API key to the .env file:<br>
	  - GROQ_API_KEY=your_api_key_here<br>

## Running the Script
  - To run the script and interact with the Groq API, execute the following command:<br>
	#### ***python taming_llm.py***

## Running the Streamlit App
  - To launch the Streamlit web interface, run:<br>
    #### ***streamlit run taming_llm.py***
  - Open your browser and navigate to the URL provided in the terminal (usually http://localhost:8501).<br>
  - Enter text in the input box and click the "Classify" button to see the results.<br>

## Code Structure
  - Main Script: taming_llm.py<br>
  - LLMClient Class: Handles interactions with the Groq API.<br>
  - classify_with_confidence Function: Classifies text and analyzes model confidence.<br>
  - compare_prompt_strategies Function: Compares different prompt strategies.<br>
  Bonus Challenges:<br>
    - calibrate_threshold: Tunes confidence thresholds based on test data.<br>
    - compare_models: Compares results across different Groq models.<br>
    - run_streamlit_app Function: Launches the Streamlit web interface.<br>



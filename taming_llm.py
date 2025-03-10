import os
import time
from dotenv import load_dotenv
import groq
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize Groq API client
class LLMClient:
    def __init__(self, model="llama3-70b-8192"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file or environment variables.")
        self.client = groq.Client(api_key=self.api_key)
        self.model = model

    def complete(self, prompt, max_tokens=1000, temperature=0.7):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

# Structured Completions
def create_structured_prompt(text, question):
    return f"""
    # Analysis Report
    ## Input Text
    {text}
    ## Question
    {question}
    ## Analysis
    """

def extract_section(completion, section_start, section_end=None):
    start_idx = completion.find(section_start)
    if start_idx == -1:
        return None
    start_idx += len(section_start)
    if section_end is None:
        return completion[start_idx:].strip()
    end_idx = completion.find(section_end, start_idx)
    if end_idx == -1:
        return completion[start_idx:].strip()
    return completion[start_idx:end_idx].strip()

# Classification with Confidence Analysis
def classify_with_confidence(client, text, categories, confidence_threshold=0.8):
    prompt = f"""
    Classify the following text into exactly one of these categories: {', '.join(categories)}.
    Response format:
    1. CATEGORY: [one of: {', '.join(categories)}]
    2. CONFIDENCE: [high|medium|low]
    3. REASONING: [explanation]
    Text to classify:
    {text}
    """
    response = client.complete(prompt)
    if not response:
        return None

    completion = response.choices[0].message.content
    category = extract_section(completion, "1. CATEGORY: ", "\n")
    confidence = extract_section(completion, "2. CONFIDENCE: ", "\n")
    reasoning = extract_section(completion, "3. REASONING: ")

    # Simulate confidence score (since logprobs is not supported)
    confidence_score = 0.9 if confidence == "high" else 0.5 if confidence == "medium" else 0.2

    if confidence_score > confidence_threshold:
        return {
            "category": category,
            "confidence": confidence_score,
            "reasoning": reasoning
        }
    else:
        return {
            "category": "uncertain",
            "confidence": confidence_score,
            "reasoning": "Confidence below threshold"
        }

# Prompt Strategy Comparison
def compare_prompt_strategies(client, texts, categories):
    strategies = {
        "basic": lambda text: f"Classify this text into one of these categories: {', '.join(categories)}",
        "structured": lambda text: f"""
        Classification Task
        Categories: {', '.join(categories)}
        Text: {text}
        Classification: """,
        "few_shot": lambda text: f"""
        Here are some examples of text classification:
        Example 1:
        Text: "The product arrived damaged and customer service was unhelpful."
        Classification: Negative
        Example 2:
        Text: "While delivery was slow, the quality exceeded my expectations."
        Classification: Mixed
        Example 3:
        Text: "Absolutely love this! Best purchase I've made all year."
        Classification: Positive
        Now classify this text:
        Text: "{text}"
        Classification: """
    }

    results = {strategy: [] for strategy in strategies.keys()}

    for strategy_name, prompt_func in strategies.items():
        for text in texts:
            prompt = prompt_func(text)
            start_time = time.time()
            response = client.complete(prompt)
            elapsed_time = time.time() - start_time

            if response:
                completion = response.choices[0].message.content
                results[strategy_name].append({
                    "text": text,
                    "completion": completion,
                    "time": elapsed_time
                })

    return results

# Bonus - Calibration Function
def calibrate_threshold(client, test_data, categories):
    """
    Tunes the confidence threshold based on test data.
    """
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_threshold = 0.8  # Default
    best_accuracy = 0

    for threshold in thresholds:
        correct = 0
        total = 0

        for text, true_label in test_data:
            result = classify_with_confidence(client, text, categories, confidence_threshold=threshold)
            if result and result["category"] == true_label:
                correct += 1
            total += 1

        accuracy = correct / total
        print(f"Threshold: {threshold}, Accuracy: {accuracy:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"Best threshold: {best_threshold} with accuracy: {best_accuracy:.2f}")
    return best_threshold

# Bonus - Compare Results Across Different Groq Models
def compare_models(client, texts, categories, models):
    """
    Compares results across different Groq models.
    """
    results = {model: [] for model in models}

    for model in models:
        client.model = model
        for text in texts:
            result = classify_with_confidence(client, text, categories)
            if result:
                results[model].append(result)

    return results

# Bonus - Web Interface with Streamlit
def run_streamlit_app():
    """
    Runs a Streamlit web interface for the classification tool.
    """
    st.title("Content Classification Tool")
    st.write("Enter text to classify its sentiment.")

    # Input text
    text = st.text_area("Input Text", "The product was amazing and delivered on time!")

    # Classify button
    if st.button("Classify"):
        with st.spinner("Classifying text..."):  # Show a loading spinner
            try:
                # Initialize Groq client
                client = LLMClient()
                categories = ["Positive", "Negative", "Mixed"]

                # Classify text
                result = classify_with_confidence(client, text, categories)
                if result:
                    st.write("Classification Result:")
                    st.json(result)
                else:
                    st.error("Failed to classify text.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Main Function
def main():
    client = LLMClient()

    # Example input
    texts = [
        "The product was amazing and delivered on time!",
        "I had a terrible experience with customer service.",
        "The quality was good, but the price was too high."
    ]
    categories = ["Positive", "Negative", "Mixed"]

    # Classification with Confidence
    for text in texts:
        result = classify_with_confidence(client, text, categories)
        print(f"Text: {text}")
        print(f"Classification: {result}")
        print()

    # Prompt Strategy Comparison
    results = compare_prompt_strategies(client, texts, categories)
    for strategy, data in results.items():
        print(f"Strategy: {strategy}")
        for item in data:
            print(f"Text: {item['text']}")
            print(f"Completion: {item['completion']}")
            print(f"Time: {item['time']:.2f}s")
            print()

    # Bonus - Calibration
    test_data = [
        ("The product was amazing and delivered on time!", "Positive"),
        ("I had a terrible experience with customer service.", "Negative"),
        ("The quality was good, but the price was too high.", "Mixed")
    ]
    best_threshold = calibrate_threshold(client, test_data, categories)
    print("Best threshold:", best_threshold)

    # Bonus - Compare Models
    models = ["llama3-70b-8192", "mixtral-8x7b-32768"]
    model_results = compare_models(client, texts, categories, models)
    for model, data in model_results.items():
        print(f"Model: {model}")
        for item in data:
            print(item)
        print()

    # Bonus - Run Streamlit App
    # run_streamlit_app()

if __name__ == "__main__":
    run_streamlit_app()
    main()
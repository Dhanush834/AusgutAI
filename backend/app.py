from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
from legal import IndianLawRAGAgent  # Import your agent class here

app = Flask(__name__)
CORS(app)  # This will allow requests from any origin

# Initialize OpenAI
openai.api_key = "your-openai-api-key"  # Replace with your OpenAI API key

# Initialize the IndianLawRAGAgent
agent = IndianLawRAGAgent(
    portkey_api_key="1+YM2sBEgaZWe45bMOq2huic1uyF",
    portkey_virtual_key="015-openai-40bada",
    bert_model_name="law-ai/InLegalBERT",
    knowledge_base_path="test.json"
)

# Define route for handling the command
@app.route('/command', methods=['POST'])
def command():
    data = request.json
    command_text = data['command']

    # Step 1: Use OpenAI to analyze and categorize the command
    openai_response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Categorize the following command into one of these options: 'answer a legal question', 'generate legal advice', 'summarize a law', 'check compliance requirements'. Then provide a detailed analysis of the context.\n\nCommand: {command_text}\n\nOptions and Analysis:",
        max_tokens=150
    )

    openai_analysis = openai_response.choices[0].text.strip()
    
    # Extract option and context from OpenAI's analysis
    option, analysis = parse_openai_analysis(openai_analysis)

    # Step 2: Based on the option, call the corresponding method
    if option == "answer a legal question":
        context = analysis  # Provide the relevant context
        answer = agent.answer_legal_question(command_text, context)
        return jsonify(message=f"Answer: {answer}")

    elif option == "generate legal advice":
        # Extract additional information from the analysis
        startup_type, situation = extract_advice_details(analysis)
        advice = agent.generate_legal_advice(startup_type, situation)
        return jsonify(message=f"Legal Advice: {advice}")

    elif option == "summarize a law":
        law_name = analysis
        summary = agent.summarize_law(law_name)
        return jsonify(message=f"Law Summary: {summary}")

    elif option == "check compliance requirements":
        startup_description = analysis
        compliance = agent.check_compliance(startup_description)
        return jsonify(message=f"Compliance Check: {compliance}")

    else:
        return jsonify(message="Invalid command or analysis error"), 400

def parse_openai_analysis(analysis_text):
    """
    Helper function to parse the OpenAI analysis into option and context.
    """
    lines = analysis_text.split('\n')
    option = lines[0].strip()
    context = "\n".join(lines[1:]).strip()
    return option, context

def extract_advice_details(analysis_text):
    """
    Extract startup type and situation from the analysis text.
    """
    # Implement your extraction logic here
    # This is a placeholder example
    parts = analysis_text.split("Situation: ")
    startup_type = parts[0].strip()
    situation = parts[1].strip() if len(parts) > 1 else ""
    return startup_type, situation

if __name__ == "__main__":
    app.run(port=5000)

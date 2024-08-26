from flask import Flask, request, jsonify
from openai import OpenAI
from flask_cors import CORS
import json
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

@app.route('/generate-questions', methods=["POST"])
def generate_questions():
    system_message = "You are an AI specialized in asking health-related questions to assess cancer risk. Your task is to create a series of personalized and easy-to-understand questions to learn about the user's habits, lifestyle, medical history, and environmental exposures. Think of these questions as something a friendly and empathetic doctor would ask. Make sure the questions are straightforward and personalized. Use simple and easy words that can be understood by a 10-year-old child. Minimize grammatical mistakes. Make sure the questions are detailed and clear. The questions should be divided into four stages: 1. **Basic Information:** Gather essential details like age, gender, location, and race/ethnicity, with options where appropriate. 2. **Health Habits:** Explore critical health behaviors, including smoking, alcohol consumption, diet, physical activity, and sleep patterns. 3. **Specific Risks:** Ask about family medical history, previous diagnoses, exposure to harmful substances (e.g., radiation, carcinogens), stress levels, and any chronic conditions. 4. **Screening and Diagnostic Tests:** Inquire about results from previous cancer screenings, findings from imaging tests (e.g., mammograms, colonoscopies), and relevant biomarker levels. Provide options only where it makes sense (like for age, gender, etc.), and use open-ended questions for more detailed responses (like lifestyle or medical history). Also, ask for the user's name to personalize the interaction. Format the response in JSON without markdown. Structure it like this: {questions: [{question: \"What is your age?\", \"type\": \"open-ended\"}, {question: \"What is your gender?\", options: [\"Male\", \"Female\", \"Other\"], \"type\": \"single-choice\"}]}"

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Generate a personalized flow of questions to assess cancer risk."}
        ],
        temperature=0.5
    )

    questions_json_str = completion.choices[0].message.content
    try:
        questions = json.loads(questions_json_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to decode JSON from the response."}), 500
    
    return jsonify(questions)

@app.route('/calculate-risk', methods=["POST"])
def calculate_risk():
    user_answers = request.json.get("user_answers", {})
    
    system_message = "You are an AI specialized in accurately assessing cancer risk based on a wide range of detailed user responses.\nYour role is to carefully analyze the user's answers concerning their habits, lifestyle, medical history, environmental exposures, and diagnostic test results. \nConsider genetic factors, lifestyle choices, demographic characteristics, environmental exposures, and results from screenings or diagnostic tests.\nProvide a nuanced risk level (very low, low, moderate, high, very high) with a precise percentage, ranging from 1% to 100%. \nThe risk level and percentage should accurately reflect the user's responses, with each factor carefully weighed.\nIf the risk is high or very high, give special attention to the seriousness of the situation but do so in a balanced and supportive manner.\nUse simple and easy words that can be understood by a 10-year-old child. Minimize grammatical mistakes. Make sure the explanation is detailed, accurate, and clear.\nGenerate the response in English.\nBegin the response with the cancer risk level and percentage."
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"User Answers: {user_answers}. Assess the cancer risk level."}
        ],
        temperature=0.5
    )
    
    risk_assessment = completion.choices[0].message.content
    return jsonify({"risk_assessment": risk_assessment})

@app.route('/personalized-guidance', methods=['POST'])
def personalized_guidance():
    cancer_risk = request.json.get("cancer_risk", "")
    user_answers = request.json.get("user_answers", {})
    
    system_message = "You are an AI that provides highly personalized health advice to reduce cancer risk.\nBased on the user's specific cancer risk assessment, their detailed answers, and the identified risk factors,\ncreate a comprehensive and actionable health plan that is truly personalized. \nFor low-risk users, suggest activities that maintain or improve their good habits, and provide encouragement.\nFor moderate-risk users, suggest actionable changes in diet, lifestyle, and screenings, with a focus on preventing escalation.\nFor high-risk users, give detailed, specific advice tailored to their unique situation. This includes targeted lifestyle changes, precise dietary advice, \nand urgent recommendations for further screenings or consultations. Acknowledge positive actions they are already taking.\nAvoid general suggestions; instead, focus on practical steps they can directly implement. Be supportive and avoid creating undue fear, but be clear about the importance of the advice.\nUse simple and easy words that can be understood by a 10-year-old child. Minimize grammatical mistakes. Make sure the guidance is clear, actionable, and highly personalized.\nIf the user has a high-risk chance, give them deeper and better guidance to improve their health.\nGenerate the response in English.\nStart the response with the cancer risk stage and percentage."
    
    prompt = f"Cancer Risk: {cancer_risk} \n User Answers: {user_answers} \n Provide personalized health guidance."
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    
    guidance = completion.choices[0].message.content
    return jsonify({"guidance": guidance})

@app.route('/summarize-text', methods=['POST'])
def summarize_text():
    user_answers = request.json.get("user_answers", {})
    
    system_message = "You are an AI specialized in summarizing user-provided health information to generate a concise, easy-to-understand summary.\nBased on the user's responses to health-related questions, create a short summary that highlights the most important details about their habits, lifestyle, medical history, and cancer risk factors.\nThe summary should be informative but concise, covering key aspects such as age, gender, health habits, specific risks, and any previous screenings or diagnostic test results.\nUse simple and easy words that can be understood by a 10-year-old child. Minimize grammatical mistakes. Make sure the summary is clear and accurate.\nGenerate the response in English."

    prompt = f"User Answers: {user_answers} \n Provide a short summary about me based on these answers."

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    summary = completion.choices[0].message.content
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
    
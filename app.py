from flask import Flask, request, render_template, session, jsonify
import numpy as np
import pandas as pd
import requests
import json
import joblib
import re
import os


app = Flask(__name__)

# Set the secret key safely
app.secret_key = os.getenv('SECRET_KEY')

# Set the API key safely
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
 
# Load the predictive model from a file
loaded_model = joblib.load('random_forest_model.pkl')
print("Model loaded successfully.")

# Define the column names for the model input
columns = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

@app.route('/next_session', methods=["GET", "POST"])
def next_session():
    """
    Route to process form data and redirect to the services page.
    """
    name = request.form['name'].capitalize()  # Capitalize the user's name
    country = request.form['country']         # Retrieve the user's country

    # Store data in session for future use
    session["name"] = name
    session["country"] = country  # Store the country in session

    # Render the services page with the user's name and country
    return render_template('services.html', country=country, name=name)

@app.route('/chat')
def chat():
    """
    Route to display the chat page.
    """
    return render_template('chat.html')   # Make sure 'chat.html' is in the templates folder



@app.route('/get_gemini_response', methods=['POST'])
def get_gemini_response():
    data = request.get_json()
    user_message = data.get('prompt', '').strip()

    if not user_message:
        return jsonify({"error": "No prompt provided."}), 400

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [{"parts": [{"text": user_message}]}]
    }

    try:
        response = requests.post(url, json=payload, headers=headers, params={"key": GEMINI_API_KEY})
        if response.status_code == 200:
            response_json = response.json()

            if 'candidates' in response_json and len(response_json['candidates']) > 0:
                bot_response = response_json['candidates'][0]['content']['parts'][0]['text']
                return jsonify({"response": bot_response})
            else:
                return jsonify({"response": "Sorry, I couldn't understand that."})
        else:
            return jsonify({"error": f"Error from Gemini API: {response.status_code}"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error making the request: {str(e)}"}), 500





def gemini_generate_content(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, json=payload, headers=headers, params={"key": GEMINI_API_KEY})
        if response.status_code != 200:
            return f"Error: Gemini API returned status code {response.status_code}. Message: {response.text}"
        response_json = response.json()
        candidates = response_json.get("candidates", [])
        if not candidates:
            return "Error: No candidates returned."
        content = candidates[0].get("content", {})
        return content.get("parts", [{}])[0].get("text", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error: Request to Gemini API failed. Details: {str(e)}"




def get_response(prompt):
    """
    Generates a chat response using Gemini API.
    """
    return gemini_generate_content(prompt)

import re

def get_predict_message(country):
    format = '''
    [
        {
            "myCountry": {"organizationName": "", "link": ""},
            "otherCountry": {"organizationName": "", "link": "", "Country": ""}
        }
    ]'''
    prompt = f"Hi, my country is {country}. Kindly give me a list of places I can get a good loan for my small business. Reply only in this JSON format without explanation: {format} Ensure links are valid and use organization names."
    raw_response = get_response(prompt)
    cleaned = re.sub(r"^```json|^```|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        parsed_json = json.loads(cleaned)
        return prompt, parsed_json
    except Exception as e:
        print("[Gemini Parsing Error]:", e)
        print("[Raw Cleaned Text]:", cleaned)
        return prompt, []




def handle_form(form_type):
    # Capture form data
    domain_interest = request.form['domain_interest']
    description = request.form['description'] if form_type == 'financial_advice' else None  # Only for financial_advice
    country_interest = request.form['country_interest']
    capital_loan = request.form['capital_loan']
    amount = request.form['amount']
    loan_pay_month = request.form['loan_pay_month']

    # Retrieve country from session (this would be set when the user goes through the next session)
    country = session.get("country", None)

    # Check if the country exists in the session (important for financial_advice and business_idea)
    if not country:
        return None, "Error: Country not found in session."

    # Generate the appropriate prompt and response based on the form_type
    if form_type == 'financial_advice':
        bot_prompt, bot_response = get_financial_advice(
            country=country,  # Pass the country
            country_interest=country_interest,
            description=description,
            capital_loan=capital_loan,
            amount=amount,
            domain_interest=domain_interest,
            loan_pay_month=loan_pay_month
        )
    elif form_type == 'business_idea':
        # Pass the country to get_business_idea
        bot_prompt, bot_response = get_business_idea(
            country=country,  # Pass the country here
            country_interest=country_interest,
            capital_loan=capital_loan,
            amount=amount,
            domain_interest=domain_interest,
            loan_pay_month=loan_pay_month
        )

    # If bot_response is invalid, log it for debugging
    if not bot_response:
        return bot_prompt, []
    
    return bot_prompt, bot_response





@app.route('/financial_advice', methods=["GET", "POST"])
def financial_advice():
    """
    Route to handle financial advice requests based on user inputs.
    """
    if request.method == "POST":
        # Call the helper function with 'financial_advice'
        bot_finance_prompt, bot_finance_response = handle_form('financial_advice')
        if isinstance(bot_finance_response, str) and bot_finance_response.startswith("Error"):
            return jsonify({"error": bot_finance_response}), 400


        # Ensure the response is valid JSON
        if bot_finance_response:
            try:
                json.dumps(bot_finance_response)  # Ensure response is serializable
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                return jsonify({"error": "Invalid JSON response received."}), 500
        else:
            return jsonify({"error": "No response received from Gemini API."}), 500

        # Store response in session
        session["bot_finance_response"] = bot_finance_response
        session["bot_finance_prompt"] = bot_finance_prompt

        # Render the page with the response
        return render_template('chat_finance.html', bot_finance_response=bot_finance_response)

    return render_template('form_financial_advice.html')

            


# Further response functions (get_further_response, get_business_idea, get_financial_advice) remain the same, just using `get_response` for interaction.




def get_further_response(prediction, question, prev_prompt, prev_response):
    """
    Generates a new prompt based on a previous conversation and a prediction result, then gets a response to it.

    This function constructs a new prompt by appending additional context based on the prediction result to 
    the previous conversation. The conversation is capped at 2500 characters for conciseness. The new prompt 
    is then used to get a further response.

    Args:
    prediction (int): The prediction result (0 for 'Yes', 1 for 'No', others for neutral).
    question (str): The new question to be asked.
    prev_prompt (str): The previous prompt in the conversation.
    prev_response (str): The previous response in the conversation.

    Returns:
    tuple: A tuple containing the new prompt and the response from get_response function.
    """

    # Combine previous prompt and response
    old = str(prev_prompt) + str(prev_response)
    previous_conv = ""
    rev_old = old[::-1]

    # Extract the last 2500 characters of the reversed conversation
    for char in rev_old:
        if len(previous_conv) < 2500:
            previous_conv += char  # Fixed missing assignment operation

    # Reverse the extracted conversation back to original order
    final_previous_conv = previous_conv[::-1]

    # Append additional text based on prediction
    if prediction == 0:  # Yes
        add_text = "again congrats on your approved loan"
    elif prediction == 1:  # No
        add_text = 'again sorry about the unapproved loan'
    else:
        add_text = ""

    final_previous_conv += add_text

    # Construct the new prompt
    new_prompt = "Question: " + question + " | Previous Context: " + final_previous_conv + " | Instruction: Provide a concise, direct answer within 800 characters."

    # Generate the response for the new prompt
    further_response = get_response(new_prompt)

    return new_prompt, further_response




@app.route('/business_idea', methods=["GET", "POST"])
def business_idea():
    """
    Route to handle business idea suggestions based on user inputs.
    """
    if request.method == "POST":
        # Safely call the form handler
        result = handle_form('business_idea')

        # Check if an error occurred
        if isinstance(result, str):
            return jsonify({"error": result}), 500

        bot_business_prompt, bot_business_response = result

        # If response is empty or invalid
        if not bot_business_response:
            return jsonify({"error": "No response received."}), 500

        # Render the response in the frontend
        return render_template('chat_business.html', bot_business_response=bot_business_response)

    # For GET request
    return render_template('form_business_idea.html')





def get_business_idea(country, country_interest, capital_loan, amount, domain_interest, loan_pay_month):
    """
    Generates a prompt for business ideas based on user's financial situation and interests, and gets a response.
    """
    format = '''
    [
        {
            "Business_Idea": "",
            "sector": "",
            "link": ""
        },
        {
            "Business_Idea": "",
            "sector": "",
            "link": ""
        }
    ]
    '''

    if capital_loan == 'capital':
        prompt = f"Hi, I'm from {country}. Kindly help curate few nice business ideas, the domain sector of the business and like to learn more on the business, considering that I have a capital of {amount} US Dollars. My domain of business interest is {domain_interest} and the country where I want to have my business is {country_interest}. Give the answer strictly in this format: {format} Thanks."
    else:
        prompt = f"Hi, I'm from {country}. Kindly help curate few nice business ideas, the domain sector of the business and like to learn more on the business, considering that I got a loan of {amount} US Dollars and I am meant to pay back in {loan_pay_month} months time. My domain of business interest is {domain_interest} and the country where I want to have my business is {country_interest}. Give the answer strictly in this format: {format} Thanks."

    
    # 🔸 Get the response from Gemini
    raw_response = get_response(prompt)

    # 🔹 Extract text
    if isinstance(raw_response, dict):
        try:
            text = raw_response['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print("[Gemini JSON structure error]:", e)
            return prompt, []
    elif isinstance(raw_response, str):
        text = raw_response
    else:
        print("[Unexpected response type]:", type(raw_response))
        return prompt, []

    # 🔹 Clean up triple backtick markdown
    cleaned = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()

    # 🔹 Parse JSON
    try:
        business_ideas = json.loads(cleaned)
        return prompt, business_ideas
    except json.JSONDecodeError as e:
        print(f"[Business Idea JSON Parsing Error]: {e}")
        print("Cleaned Text That Failed Parsing:", cleaned)
        return prompt, []
    



def get_financial_advice(country, country_interest, description, capital_loan, amount, domain_interest, loan_pay_month):
    """
    Generates a prompt for financial advice and gets a structured JSON response.
    """
    format = '''
    {
        "financial_breakdown": "",
        "link": ""
    }
    '''

    # Construct prompt
    if capital_loan == 'capital':
        prompt = f"Hi, I'm from {country}. Kindly help curate a comprehensive financial breakdown with link to read more on it, for how I would manage my business considering that I have a capital of {amount} US Dollars. My domain of business interest is {domain_interest}, the description is: {description}, and the country where I want to have my business is {country_interest}. Make your answer strictly in this format: {format}."
    else:
        prompt = f"Hi, I'm from {country}. Kindly help curate a comprehensive financial breakdown with link to read more on it, for how I would manage my business considering that I got a loan of {amount} US Dollars and I am meant to pay back in {loan_pay_month} months time. My domain of business interest is {domain_interest}, the description is: {description}, and the country where I want to have my business is {country_interest}. Make your answer strictly in this format: {format}."

    # Call Gemini API
    advice_response = get_response(prompt)

    # Check for error in response
    if isinstance(advice_response, str) and advice_response.startswith("Error:"):
        print("[Gemini Error]:", advice_response)
        return prompt, []

    # Extract actual text
    if isinstance(advice_response, dict):
        try:
            text = advice_response['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print("[Gemini Financial JSON structure error]:", e)
            return prompt, []
    else:
        text = advice_response

    # 🔶 Clean up any ```json markdown blocks
    cleaned = re.sub(r"^```json|^```|```$", "", text.strip(), flags=re.MULTILINE).strip()

    # 🔶 Try parsing JSON safely
    try:
        financial_data = json.loads(cleaned, strict=False)  # use strict=False to allow minor control chars
        return prompt, financial_data
    except json.JSONDecodeError as e:
        print(f"[Financial Advice JSON Parsing Error]: {e}")
        print("Cleaned Text That Failed Parsing:", cleaned)
        return prompt, []




@app.route('/chat_predict', methods=["GET", "POST"])
def chat_predict():
    if request.method == "POST":
        depend = request.form['depend']
        education = request.form['education']
        employment = request.form['employment']
        marital_status = request.form['marital_status']
        property_area = request.form['property_area']
        loan_purpose = request.form['loan_purpose']
        email = request.form.get('email')
        phone = request.form.get('phone')
        income = request.form['income']
        loan_amount = request.form['loan_amount']
        loan_term = request.form['loan_term']
        score = request.form['score']
        resident = request.form['resident']
        commercial = request.form['commercial']
        luxury = request.form['luxury']
        bank = request.form['bank']

        prediction_data = {
            'no_of_dependents': depend,
            'education': education,
            'self_employed': employment,
            'income_annum': income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': score,
            'residential_assets_value': resident,
            'commercial_assets_value': commercial,
            'luxury_assets_value': luxury,
            'bank_asset_value': bank
        }

        arr = pd.DataFrame([prediction_data])
        pred = int(loaded_model.predict(arr)[0])

        name = session.get("name", "User")
        country = session.get("country", "your country")

        # ✅ FIXED: Directly get parsed response, no double json.loads
        prompt, bot_predict_response = get_predict_message(country)

        session["pred"] = pred
        session["bot_predict_prompt"] = prompt
        session["bot_predict_response"] = bot_predict_response

        return render_template(
            'chat_predict.html',
            pred=pred,
            email=email,
            phone=phone,
            purpose=loan_purpose,
            name=name,
            country=country,
            bot_predict_response=bot_predict_response
        )

    return render_template('form_predict.html')

model= None

@app.route('/', methods=["GET", "POST"])
def main():
    """
    Route for the main page of the application.

    This route handles both GET and POST requests and renders the 'index.html' template,
    which is typically the homepage or landing page of the application.

    Returns:
    render_template: Renders the 'index.html' template.
    """
    return render_template('index.html')



@app.route('/form_predict', methods=["GET", "POST"])
def form_predict():
    """
    Route for the prediction form page.

    This route renders the 'form_predict.html' template, which usually contains a form
    for users to input data for predictions.

    Returns:
    render_template: Renders the 'form_predict.html' template.
    """
    return render_template('form_predict.html')



@app.route('/form_business_idea', methods=["GET", "POST"])
def form_business_idea():
    """
    Route for the business idea form page.

    This route renders the 'form_business_idea.html' template, where users can input information
    to get suggestions or advice on business ideas.

    Returns:
    render_template: Renders the 'form_business_idea.html' template.
    """
    return render_template('form_business_idea.html')



@app.route('/sign_in', methods=["GET", "POST"])
def sign_in():
    """
    Route for the sign-in page.

    This route renders the 'sign_in.html' template, which typically contains a form for
    user authentication (login).

    Returns:
    render_template: Renders the 'sign_in.html' template.
    """
    return render_template('sign_in.html')
       

@app.route('/services', methods=["GET", "POST"])
def services():
    """
    Route for the services page.

    This route renders the 'services.html' template, which typically lists the services
    or features offered by the application.

    Returns:
    render_template: Renders the 'services.html' template.
    """
    return render_template('services.html')



@app.route('/form_financial_advice', methods=["GET", "POST"])
def form_financial_advice():
    """
    Route for the financial advice form page.

    This route renders the 'form_financial_advice.html' template, where users can input details
    to receive financial advice or information.

    Returns:
    render_template: Renders the 'form_financial_advice.html' template.
    """
    return render_template('form_financial_advice.html')





@app.route('/further_predict_chat', methods=["GET", "POST"])
def further_predict_chat():
    """
    Route to handle further prediction interactions in a chat interface.

    This route retrieves the previous prediction result and related conversation context from the session.
    If a new POST request is made, it processes the user's question and gets a further response based on 
    the previous context and prediction. The new response is then stored in the session and sent back to 
    the user in JSON format.

    Returns:
    jsonify: A JSON response containing the prediction response for the user's question.
    """
    ...
    predict_response = ""  # Add this at the top to avoid reference error
    ...

    # Retrieve previous prediction and conversation context from session
    pred = session.get("pred", None)
    bot_predict_prompt = session.get("bot_predict_prompt", None)
    bot_predict_response = session.get("bot_predict_response", None)

    # Process new question and get further response if method is POST
    if request.method == 'POST':
        predict_question = request.form['question']

        # Get further response based on the new question and previous context
        predict_prompt, predict_response = get_further_response(prediction=pred, question=predict_question,
                                                                prev_prompt=bot_predict_prompt, prev_response=bot_predict_response)

        # Update session with new response and prompt
        session["bot_predict_response"] = predict_response
        session["bot_predict_prompt"] = predict_question

    # Return the new response in JSON format
    return jsonify({"response": predict_response })


@app.route('/further_business_chat', methods=["GET", "POST"])
def further_business_chat():
    """
    Route to handle further interactions in the business chat interface.

    This route retrieves the previous business chat response and related conversation context from the session.
    If a new POST request is made, it processes the user's question and gets a further response based on 
    the previous context. The new response is then stored in the session and sent back to the user in JSON format.

    Returns:
    jsonify: A JSON response containing the further response for the user's business-related question.
    """

    # Retrieve previous business chat response and prompt from session
    bot_business_response = session.get("bot_business_response", None)
    bot_business_prompt = session.get("bot_business_prompt", None)

    # Process new question and get further response if method is POST
    if request.method == 'POST':
        business_question = request.form['question']

        # Get further response based on the new question and previous context
        business_prompt, business_response = get_further_response(prediction="", question=business_question,
                                                                  prev_prompt=bot_business_prompt, prev_response=bot_business_response)

        # Update session with new response and prompt
        session["bot_business_response"] = business_response
        session["bot_business_prompt"] = business_question

    # Return the new response in JSON format
    return jsonify({"response": business_response })
       


@app.route('/further_finance_chat', methods=["GET", "POST"])
def further_finance_chat():
    """
    Route to handle follow-up interactions in the financial chat interface.

    This route retrieves the previous financial advice response and related conversation context 
    from the session. If a new POST request is made, it processes the user's financial question 
    and gets a further response based on the previous context. The new response is then stored 
    in the session under the correct keys and sent back to the user in JSON format.

    Returns:
    jsonify: A JSON response containing the further response for the user's finance-related question.
    """

    # Retrieve previous financial advice response and prompt from session
    bot_finance_response = session.get("bot_finance_response", None)
    bot_finance_prompt = session.get("bot_finance_prompt", None)

    # Process new question and get further response if method is POST
    if request.method == 'POST':
        finance_question = request.form['question']

        # Get further response based on the new question and previous context
        finance_prompt, finance_response = get_further_response(prediction="", question=finance_question,
                                                                prev_prompt=bot_finance_prompt, prev_response=bot_finance_response)

        # Update session with new response and prompt
        session["bot_finance_response"] = finance_response
        session["bot_finance_prompt"] = finance_question

    # Return the new response in JSON format
    return jsonify({"response": finance_response })



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
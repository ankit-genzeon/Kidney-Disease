from flask import Flask, render_template, request, make_response
import PyPDF2
import pandas as pd
import pickle
import re
import io

app = Flask(__name__)

# Load the trained model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Load the scaler from the pickle file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    pdf_file = request.files['file']
    if pdf_file:
        # Load the PDF file
        reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from each page
        text = ''
        for page in reader.pages:
            text += page.extract_text()

        # Preprocess the extracted text to obtain the feature values
        feature_values = {}
        pattern = r'(\w+)\s*=\s*([\d.]+)'
        matches = re.findall(pattern, text)
        for match in matches:
            feature_name = match[0].lower()
            feature_value = float(match[1])
            feature_values[feature_name] = feature_value

        # Convert the feature values into a DataFrame
        df = pd.DataFrame([feature_values])

        # Apply feature scaling on the feature values
        df_scaled = scaler.transform(df)

        # Make predictions on the feature values
        predictions = classifier.predict(df_scaled)

        # Render the result template with the predictions and feature_values
        return render_template('result.html', prediction=predictions[0], feature_values=feature_values)

    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    prediction = request.form['prediction']
    feature_values = {
        'Age': request.form['age'],
        'Blood Pressure': request.form['bp'],
        'Specific Gravity': request.form['sg'],
        'Albumin': request.form['al'],
        'Sugar': request.form['su'],
        'Blood Glucose Random': request.form['bgr'],
        'Blood Urea': request.form['bu'],
        'Serum Creatinine': request.form['sc'],
        'Sodium': request.form['sod'],
        'Potassium': request.form['pot'],
        'Hemoglobin': request.form['hemo'],
        'Packed Cell Volume': request.form['pcv'],
        'White Blood Cell Count': request.form['wc'],
        'Red Blood Cell Count': request.form['rc']
    }
    
    # Generate PDF report with the data and prediction
    report = generate_report(prediction, feature_values)

    # Create a response with the PDF file
    response = make_response(report)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=report.pdf'
    return response

def generate_report(prediction, feature_values):
    # Create a new PDF writer object
    writer = PyPDF2.PdfWriter()

    # Create a new page for the report
    report_page = PyPDF2.pdf.PageObject()

    # Set the content for the report page
    report_text = f"Prediction: {prediction}\n\n"
    for feature, value in feature_values.items():
        report_text += f"{feature}: {value}\n"

    # Set the content for the report page
    report_page.mergePage(PyPDF2.pdf.PageObject.create_text_object(report_text))

    # Add the report page to the PDF writer
    writer.addPage(report_page)

    # Create a bytes buffer to store the PDF content
    report_buffer = io.BytesIO()

    # Write the PDF content to the buffer
    writer.write(report_buffer)

    # Seek to the beginning of the buffer
    report_buffer.seek(0)

    # Return the PDF report as bytes
    return report_buffer.getvalue()


if __name__ == '__main__':
    app.run()

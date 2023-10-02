"""
This app uses the YASA package to perform sleep staging on an EDF file.

Vallat, Raphael, and Matthew P. Walker. “An open-source, high-performance tool for automated sleep staging.” Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092
"""

from flask import Flask, request, render_template, flash, redirect, url_for
import mne
import yasa
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from jinja2 import Environment, FileSystemLoader

# Create a folder named 'uploads' in the same directory as your app.py
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'SleepAnalyzerSecretKey'  


def custom_round(value, precision=2):
    if isinstance(value, (int, float)):
        return round(value, precision)
    return value

app.jinja_env.filters['custom_round'] = custom_round

# Define a function to load the EDF file
def load_edf(filename):
    # Read in the EDF file
    raw = mne.io.read_raw_edf(filename, preload=True)
    return raw

# Define a function to preprocess the EDF file
def preprocess(raw):
    # Resample and filter the raw data
    raw.resample(100)
    raw.filter(0.3, 45)
    return raw

# Define a function to perform sleep staging
def sleep_staging(raw, selected_channel):
    # Convert the selected_channel to lowercase for case-insensitive matching
    selected_channel_lower = selected_channel.lower()

    # Find a matching channel name
    matching_channel = next((ch for ch in raw.ch_names if selected_channel_lower in ch.lower()), None)

    if matching_channel:
        sls = yasa.SleepStaging(raw, eeg_name=matching_channel)
        # Predict the sleep stages
        hypno_pred = sls.predict()
        hypno_pred = yasa.hypno_str_to_int(hypno_pred)
        return hypno_pred, None
    else:
        error_message = f'Error: EEG channel "{selected_channel}" not found.'
        flash(error_message, 'error')  # Flash an error message
        return None, error_message



def generate_sleep_stage_percentage_chart(sleep_statistics):
    # Extract sleep stage percentages from the sleep statistics dictionary
    percentages = {
        'N1': sleep_statistics['%N1'],
        'N2': sleep_statistics['%N2'],
        'N3': sleep_statistics['%N3'],
        'REM': sleep_statistics['%REM'],
    }

    # Create a pie chart
    labels = percentages.keys()
    values = percentages.values()
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Save the chart to a temporary file or buffer
    chart_buffer = BytesIO()
    plt.savefig(chart_buffer, format='png')
    plt.close()  # Close the chart to release resources
    chart_buffer.seek(0)

    # Return the base64-encoded image
    chart_base64 = base64.b64encode(chart_buffer.read()).decode('utf-8')
    return chart_base64


@app.route('/', methods=['GET', 'POST'])
def index():
    raw_ch_names = ['C3-A2', 'O2-A1', 'C4-A1', 'O1-A2', 'Fp1-A2', 'Fp2-A1', 'F7-A2', 'F3-A2', 'FZ-A2', 'F4-A1', 'F8-A1', 'T3-A2', 'CZ-A2', 'T4-A1', 'T5-A2', 'P3-A2', 'PZ-A2', 'P4-A1', 'T6-A1', 'Fpz-Cz']

    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        
        # Save the file to disk
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # Load the EDF file
        raw = load_edf(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Extract the channel names from the loaded EDF file
        raw_ch_names = raw.ch_names

        # Get the selected EEG channel from the dropdown
        selected_channel = request.form['channel']
        
        # Preprocess the raw data
        raw = preprocess(raw)

        # Perform sleep staging
        hypno_pred, error = sleep_staging(raw, selected_channel=selected_channel)

        if error:
            # Flash the error message
            flash(error, 'error')
            # Redirect back to the index page with the flashed error message
            return redirect(url_for('index'))

        # Calculate sleep statistics
        sleep_statistics = yasa.sleep_statistics(hypno_pred, sf_hyp=1/30)

        
        # Plot the hypnogram
        fig, ax = plt.subplots()
        yasa.plot_hypnogram(hypno_pred);

        # Create a temporary in-memory buffer to save the figure
        img_data = BytesIO()
        fig.savefig(img_data, format='png')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.read()).decode('utf-8')

        # Close the figure to release resources
        plt.close(fig)

        # Determine the selected display option
        display_option = request.form.get('display_option')

        if display_option == 'sleep_stats':
            # Sleep stage percentage chart
            sleep_stage_percentage_chart = generate_sleep_stage_percentage_chart(sleep_statistics)
            return render_template('result.html', sleep_stats=sleep_statistics, sleep_stage_percentage_img=sleep_stage_percentage_chart, display_option=display_option)
        elif display_option == 'hypnogram':
            return render_template('result.html', img_data=img_base64, display_option=display_option)
        elif display_option == 'both':
            # Sleep stage percentage chart
            sleep_stage_percentage_chart = generate_sleep_stage_percentage_chart(sleep_statistics)
            return render_template('result.html', img_data=img_base64, sleep_stats=sleep_statistics, sleep_stage_percentage_img=sleep_stage_percentage_chart, display_option=display_option)

    return render_template('index.html', raw_ch_names=raw_ch_names)

if __name__ == '__main__':
    app.run(debug=True)

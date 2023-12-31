# SleepAnalyzer
#### Video Demo: https://youtu.be/k9g1fUVLvQg
#### Description:

Curious about what your sleep patterns reveal about you? You're at the right place!

With this SleepAnalyzer tool, you can upload your polysomnography (PSG) recording and get instant insights into your sleep quality and patterns.

**Easy to use**: Just upload your EDF file and choose what kind of analysis you want.

**Comprehensive analysis**: Get basic sleep statistics and/or a detailed hypnogram.

**Fast and reliable**: Our tool uses artificial intelligence for quick and accurate results.

Whether you're a patient, clinician, researcher, or just curious about sleep, SleepAnalyzer is here to help you achieve better sleep health.

### app.py

`app.py` is the main script of the SleepAnalyzer web application. It utilizes the Flask framework to create a web interface for analyzing sleep patterns from polysomnography (PSG) data in EDF format. The primary functionalities and components of this script include:

- **Dependencies:** Importing necessary libraries such as Flask, MNE, YASA, and Matplotlib for handling web requests, EEG data, sleep staging, and result visualization.

- **Configuration:** Setting up configuration variables like the `UPLOAD_FOLDER` for storing uploaded EDF files and a secret key for Flask sessions.

- **Custom Filter:** Defining a custom Jinja2 filter `custom_round` to round numeric values for display.

- **Data Loading and Preprocessing:** Implementing functions to load and preprocess the EDF data. The data is resampled and filtered to prepare it for sleep staging.

- **Sleep Staging:** Utilizing the YASA package to predict sleep stages based on the EEG data. The user selects an EEG channel, and the script performs sleep staging on that channel.

- **Result Visualization:** Generating a hypnogram and sleep stage percentage chart based on the sleep staging results. These visualizations help users interpret their sleep patterns.

- **Web Routes:** Defining web routes for handling user interactions. The main route (`'/'`) allows users to upload EDF files, select EEG channels, choose display options (basic sleep statistics, hypnogram, or both), and initiate the analysis.

- **Result Rendering:** Rendering the analysis results to HTML templates, including the hypnogram, sleep statistics, and sleep stage percentage chart.

### Index.html

This HTML file represents the main page of the SleepAnalyzer web application. It's the initial page where users can upload their EDF files and configure analysis options. Here's a breakdown of its components:

- **Header and Banner Image:** The header section contains a banner image and a welcoming title. The banner image adds visual appeal to your application and sets the mood.

- **Introduction:** Beneath the header, you have an introduction section that explains the purpose of SleepAnalyzer. It highlights key features, including ease of use, comprehensive analysis, and reliability. It encourages users to explore their sleep data.

- **File Upload Form:** The main content area of the page features a form for uploading EDF files. Users can choose an EEG channel for analysis and select what kind of analysis they want to perform (basic sleep statistics, hypnogram, or both).

- **Loading Spinner:** A loading spinner is included to indicate that the page is loading. It's hidden by default but appears when the analysis is initiated.

### Result.html

This HTML file represents the result page of the SleepAnalyzer web application. Users are redirected to this page after submitting their analysis request. Here's an explanation of its contents:

- **Result Title:** The page displays the title "Sleep Analysis Result" to indicate the nature of the content.

- **Hypnogram Section (Conditional):** If the user chose to display the hypnogram as part of their analysis, this section appears. It provides an explanation of the hypnogram, its significance, and what a typical night's sleep cycle looks like. An image of the hypnogram generated from the analysis is included.

- **Sleep Statistics Section (Conditional):** If the user chose to display sleep statistics, this section appears. It provides detailed information about various sleep parameters, such as Time in Bed (TIB), Sleep Period Time (SPT), Wake After Sleep Onset (WASO), Total Sleep Time (TST), Sleep Efficiency (SE), and more. Each parameter is described, and the corresponding value is displayed.

- **Sleep Stage Percentage Chart:** Regardless of whether the user selected to display the hypnogram or sleep statistics, a sleep stage percentage chart is always included. This chart visualizes the distribution of sleep stages (N1, N2, N3, REM) as a percentage of Total Sleep Time (TST). A brief explanation of each sleep stage is provided below the chart.

- **Toggle Buttons and Descriptions:** For each sleep parameter in the sleep statistics section, a toggle button allows users to expand or collapse additional descriptions. This feature enhances the user experience by providing more information on each parameter.

### style.css

The `style.css` file is responsible for styling the SleepAnalyzer web application's user interface. It governs the appearance and layout of various elements on the application's web pages, ensuring an attractive and user-friendly design. 

#### Credits:

This project uses the YASA package to perform sleep staging on an EDF file. The user can upload an EDF file that contains the different signals of their polysomnography (PSG) recording.

Vallat, Raphael, and Matthew P. Walker. “An open-source, high-performance tool for automated sleep staging.” Elife 10 (2021). doi: https://doi.org/10.7554/eLife.70092

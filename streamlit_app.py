import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd
import plotly.express as px
import requests

# Load the model (uncomment when you have your model file)
model = joblib.load('kmens_model.joblib')
scaler = joblib.load('kmens_scaler.joblib')

# provider_dict = {'Advancing Women in Tech': 0, 'Akamai Technologies, Inc.': 1, 'Amazon Web Services': 2, 'American Museum of Natural History': 3, 'American Psychological Association': 4, 'Aptly': 5, 'Arizona State University': 6, 'Arm': 7, 'Atlassian': 8, 'Autodesk': 9, 'Automatic Data Processing, Inc. (ADP)': 10, 'Automation Anywhere': 11, 'Berklee': 12, 'Birla Institute of Technology & Science, Pilani': 13, 'Board Infinity': 14, 'CVS Health': 15, 'California Institute of the Arts': 16, 'Caltech': 17, 'Case Western Reserve University': 18, 'CertNexus': 19, 'Cisco Learning and Certifications': 20, 'Cloudera': 21, 'Codio': 22, 'Columbia University': 23, 'Commonwealth Education Trust': 24, 'CompTIA': 25, 'Copenhagen Business School': 26, 'Corporate Finance Institute': 27, 'Coursera': 28, 'Coursera Instructor Network': 29, 'Dartmouth College': 30, 'Databricks': 31, 'Deep Teaching Solutions': 32, 'DeepLearning.AI': 33, 'Dell': 34, 'Digital Marketing Institute': 35, 'Duke University': 36, 'EC-Council': 37, 'EDHEC Business School': 38, 'EDUCBA': 39, 'EIT Digital': 40, 'ESSEC Business School': 41, 'Edge Impulse': 42, 'Edureka': 43, 'Eindhoven University of Technology': 44, 'Emory University': 45, 'Erasmus University Rotterdam': 46, 'Fortinet, Inc.': 47, 'Fractal Analytics': 48, 'Fundação Instituto de Administração': 49, 'Genentech': 50, 'Georgia Institute of Technology': 51, 'Goodwill Industries International': 52, 'Google': 53, 'Google AR & VR': 54, 'Google Cloud': 55, 'HEC Paris': 56, 'HRCI': 57, 'Hebrew University of Jerusalem': 58, 'Howard University': 59, 'HubSpot Academy': 60, 'IBM': 61, 'IE Business School': 62, 'IESE Business School': 63, 'IIMA - IIM Ahmedabad': 64, 'INSEAD': 65, 'ISAE-SUPAERO': 66, 'ISC2': 67, 'Icahn School of Medicine at Mount Sinai': 68, 'Illinois Tech': 69, 'Imperial College London': 70, 'Indian Institute for Human Settlements': 71, 'Indian School of Business': 72, 'Infosec': 73, 'Institut Mines-Télécom': 74, 'Institute for the Future': 75, 'Intel': 76, 'Interactive Brokers': 77, 'Intuit': 78, 'Johns Hopkins University': 79, 'Keller Williams': 80, 'Kennesaw State University': 81, 'Khalifa University': 82, 'Knowledge Accelerators': 83, 'Korea Advanced Institute of Science and Technology(KAIST)': 84, 'L&T EduTech': 85, 'LearnKartS': 86, 'LearnQuest': 87, 'London Business School': 88, 'Ludwig-Maximilians-Universität München (LMU)': 89, 'Lund University': 90, 'Macquarie University': 91, 'MathWorks': 92, 'McMaster University': 93, 'MedCerts': 94, 'Meta': 95, 'Michigan State University': 96, 'Microsoft': 97, 'Multiple educators': 98, 'NVIDIA': 99, 'Nanjing University': 100, 'Nanyang Technological University, Singapore': 101, 'National Academy of Sports Medicine': 102, 'National Taiwan University': 103, 'National University of Singapore': 104, 'New York Institute of Finance': 105, 'New York University': 106, 'Northeastern University': 107, 'Northwestern University': 108, 'Novartis': 109, 'O.P. Jindal Global University': 110, 'Olay': 111, 'Oracle': 112, 'Palo Alto Networks': 113, 'Parsons School of Design, The New School': 114, 'Peking University': 115, 'Pohang University of Science and Technology(POSTECH)': 116, 'Politecnico di Milano': 117, 'Pontificia Universidad Católica de Chile': 118, 'PwC': 119, 'PwC India': 120, 'Queen Mary University of London': 121, 'Red Hat': 122, 'Rice University': 123, 'Royal Holloway, University of London': 124, 'Rutgers the State University of New Jersey': 125, 'SAP': 126, 'SAS': 127, 'Salesforce': 128, 'Sciences Po': 129, 'Scrimba': 130, 'Shanghai Jiao Tong University': 131, 'Siemens': 132, 'SkillUp EdTech': 133, 'SoFi': 134, 'Splunk Inc.': 135, 'Stanford University': 136, 'Starweaver': 137, 'Sungkyunkwan University': 138, 'Tableau Learning Partner': 139, 'Tally Education and Distribution Services Private Limited': 140, 'Technical University of Denmark (DTU)': 141, 'Technical University of Munich (TUM)': 142, 'Technion - Israel Institute of Technology': 143, 'Tecnológico de Monterrey': 144, 'Tel Aviv University': 145, 'The Chinese University of Hong Kong': 146, 'The George Washington University': 147, 'The Hong Kong University of Science and Technology': 148, 'The Linux Foundation': 149, 'The Museum of Modern Art': 150, 'The Pennsylvania State University': 151, 'The State University of New York': 152, 'The University of Chicago': 153, 'The University of Edinburgh': 154, 'The University of Hong Kong': 155, 'The University of Melbourne': 156, 'The University of North Carolina at Chapel Hill': 157, 'The University of Sydney': 158, 'The University of Tokyo': 159, 'Tsinghua University': 160, 'UNSW Sydney (The University of New South Wales)': 161, 'UiPath': 162, 'Unilever': 163, 'Universidad de Palermo': 164, 'Universitat Autònoma de Barcelona': 165, 'Universitat de Barcelona': 166, 'Universiteit Leiden': 167, 'University at Buffalo': 168, 'University of Alberta': 169, 'University of Amsterdam': 170, 'University of Arizona': 171, 'University of California San Diego': 172, 'University of California, Davis': 173, 'University of California, Irvine': 174, 'University of California, Santa Cruz': 175, 'University of Cape Town': 176, 'University of Colorado Boulder': 177, 'University of Colorado System': 178, 'University of Copenhagen': 179, 'University of Florida': 180, 'University of Geneva': 181, 'University of Glasgow': 182, 'University of Houston': 183, 'University of Illinois Urbana-Champaign': 184, 'University of Lausanne': 185, 'University of Leeds': 186, 'University of London': 187, 'University of Manchester': 188, 'University of Maryland, College Park': 189, 'University of Michigan': 190, 'University of Minnesota': 191, 'University of New Mexico': 192, 'University of North Texas': 193, 'University of Pennsylvania': 194, 'University of Pittsburgh': 195, 'University of Rochester': 196, 'University of Toronto': 197, 'University of Virginia': 198, 'University of Virginia Darden School Foundation': 199, 'University of Washington': 200, 'University of Western Australia': 201, 'University of Zurich': 202, 'Università Bocconi': 203, 'Università di Napoli Federico II': 204, 'Utrecht University': 205, 'Vanderbilt University': 206, 'Voxy': 207, 'Wesleyan University': 208, 'West Virginia University': 209, 'Whizlabs': 210, 'Yale University': 211, 'Yonsei University': 212, 'École Polytechnique': 213, 'École Polytechnique Fédérale de Lausanne': 214, 'École des Ponts ParisTech': 215}
# level_dict = {'Advanced': 0, 'Beginner': 1, 'Intermediate': 2, 'Mixed': 3}
# course_type_dict = {'Course': 0, 'Professional Certificate': 1, 'Specialization': 2}
# duration_weeks_dict = {'1 - 4': 0, '12 - 24': 1, '4 - 12': 2}

# Streamlit layout with sidebar menu
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "A Guide to Coursera's Premier Data Courses", "K-Means Clustering", "Prediction"],
        icons=["house", "file-earmark-text", "file-earmark-text"],
        menu_icon="cast",
        default_index=0,
    )

def get_readable_result(prediction):
    result = {
                0:"Not a popular course, not recommended", 
                1: "This is a popular course, recommended!", 
                2: "Average popularity, could be useful",
                }
    
    return result[prediction]

if selected == "Home":
    st.title("Welcome to Home Page")
    st.write("This is the home page.")

elif selected == "A Guide to Coursera's Premier Data Courses":
    st.title("Coursera Data Analysis")
    
    st.subheader("Univariate Analysis")
    st.image("chart1.png")
    st.write("Most Coursera courses have ratings between 4 and 5, indicating that it is a valuable platform for taking courses.")
    st.image("chart2.png")
    st.write("IBM demonstrates its expertise by offering courses across a wide range of tracks.")
    st.image("chart3.png")
    st.write("We can indicate that most Coursera users are beginners.")
    st.image("chart4.png")
    st.write("Most of the course ratings fall between 0 and 50k.")
    st.image("chart5.png")
    st.write("Since most users are beginners, the courses have the highest enrollment numbers.")
    st.image("chart6.png")
    st.write("Users prefer to take courses that do not exceed three months in duration.")
    
    st.subheader("Bivariate/Multivariate Analysis")
    st.write("**Chart 1: Level vs. Type**")
    st.image("chart7.png")
    st.write("The chart shows a dominant focus on 'Beginner' level courses in 'Course' and 'Specialization' types.")
    st.image("chart8.png")
    st.write("Most IBM enrollees have liked the courses.")
    st.image("chart9.png")
    st.write("The 'Python for Everybody' course has the highest number of reviews, indicating its popularity among learners compared to other courses.")
    st.image("chart10.png")
    st.write("A three-month period is the most preferred duration for courses.")
    st.image("chart11.png")
    st.write("Professional certificates are not offered by many companies.")
    st.image("chart12.png")
    st.write("Most of the top providers by reviews are universities.")
    st.image("chart12.png")
    st.write("The majority of courses on Coursera have high ratings.")

elif selected == "K-Means Clustering":
    st.title("K-Means Clustering")
    st.image("Silhouette Score.png")
    st.image("Silhouette plot for the various cluster.png")
    st.image("cluster_non_PCA.png")
    st.image("Density.png")
    st.image("distribution of clusters by provider.png")

elif selected == "Prediction":
    st.title("Prediction Page")
    
    with st.form("prediction_form"):
        rating = st.number_input("Rating (1-5):", min_value=0.0, max_value=5.0)
        reviews = st.number_input("Total Reviews:", min_value=0)
        duration_weeks = st.number_input("Duration in weeks:", min_value=1.0)

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:

        # Prepare data to send to FastAPI
        cors_data = {
            "rating": rating,
            "reviews": reviews,
           "duration_weeks": duration_weeks
        }

        try:
            # Send data to FastAPI
            response = requests.post("https://team-pirates-project-5.onrender.com/predict/", json=cors_data)
            response.raise_for_status()  # Will raise an HTTPError for bad responses

            # Extract and display prediction
            prediction = response.json().get("prediction", "No prediction found")

            prediction = get_readable_result(prediction)

            st.write(f"Prediction: {prediction}")
            
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

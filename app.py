import numpy as np

from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl','rb'))
@app.route('/')
def h():
    return render_template('home.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/index')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['GET'])
def predict():
    def safe_get(name, default='0'):
        val = request.args.get(name, default)
        return val.strip() if val else default

    name = safe_get('name', 'Candidate')
    cgpa = safe_get('cgpa')
    projects = safe_get('projects')
    workshops = safe_get('workshops')
    mini_projects = safe_get('mini_projects')
    skills = safe_get('skills')
    communication_skills = safe_get('communication_skills')
    internship = safe_get('internship')
    hackathon = safe_get('hackathon')
    tw_percentage = safe_get('tw_percentage')
    te_percentage = safe_get('te_percentage')
    backlogs = safe_get('backlogs')

    # Calculate number of skills
    s = len([skill.strip() for skill in skills.split(',') if skill.strip()])

    # First model (placement prediction)
    input_features = [cgpa, projects, workshops, mini_projects, s,
                      communication_skills, internship, hackathon,
                      tw_percentage, te_percentage, backlogs]
    input_array = np.asarray(input_features, dtype=float)
    placement_result = model.predict([input_array])[0]

    placed_flag = '1' if placement_result == 'Placed' else '0'

    # Second model (salary prediction)
    salary_input = input_features + [placed_flag]
    salary_array = np.asarray(salary_input, dtype=float)
    predicted_salary = model1.predict([salary_array])[0]
    formatted_salary = "{:,.0f}".format(predicted_salary)

    # Output messages
    if placement_result == 'Placed':
        out = f'Congratulations {name}! You have high chances of getting placed!'
        out2 = f'Your expected salary is INR {formatted_salary} per annum.'
    else:
        out = f'Sorry {name}, you have low chances of getting placed. All the best!'
        out2 = 'Try to improve your skills and profile.'

    return render_template('output.html', output=out, output2=out2)
     

if __name__ == "__main__":
    app.run(debug=True)
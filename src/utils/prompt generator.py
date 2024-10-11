import pandas as pd

# Data for each category and examples in CSV format
data = {
    "email_class": ["Student Inquiry"] * 10 + ["Academic Collaboration Inquiry"] * 10 + ["Corporate Inquiry"] * 10,
    "category": [
        # Student Inquiry Categories
        "Request for Course Syllabus", 
        "Query on Assignment Deadlines", 
        "Inquiry on Exam Schedule and Format", 
        "Request for Office Hours Information", 
        "Request for Tutoring Resources", 
        "Question on Lecture or Lab Schedule", 
        "Appeal for Grade Review", 
        "Inquiry on Graduation Requirements", 
        "Request for Research Project Collaboration", 
        "Request for Access to Study Materials",

        # Academic Collaboration Inquiry Categories
        "Research Collaboration Proposal", 
        "Access to Shared Research Data", 
        "Request for Joint Grant Submission", 
        "Paper Co-authorship Request", 
        "Invitation to Present at Conference", 
        "Request for Research Tool Sharing", 
        "Invitations for Workshops and Seminars", 
        "Data Sharing Agreement Request", 
        "Inquiry on Ethics Submission", 
        "Proposal for Joint Review Paper",

        # Corporate Inquiry Categories
        "Internship Opportunities", 
        "Proposal for Partnership", 
        "Request for Research Sponsorship", 
        "Recruitment for AI Roles", 
        "Corporate Training Requests", 
        "Product Licensing Inquiries", 
        "Consulting Service Requests", 
        "Confidential Partnership Discussions", 
        "Request for Guest Lectures", 
        "Feedback on Corporate Research"
    ],
    "example_subject": [
        # Student Inquiry Examples
        "Request for Syllabus for 'Introduction to AI'", 
        "When is the Neural Networks Assignment Due?", 
        "When is the Midterm Exam for 'NLP'?", 
        "What are the Office Hours for Prof. Jane Doe?", 
        "Are Tutoring Sessions Available for 'Advanced ML'?", 
        "Has the Lab Session for 'Intro to AI' Been Rescheduled?", 
        "How Can I Appeal My Grade for 'NLP'?", 
        "How Many Credits are Needed for Graduation?", 
        "Can I Join the 'AI for Healthcare Diagnostics' Project?", 
        "Can You Share the Reading Material for 'Week 1' of 'Advanced ML'?",

        # Academic Collaboration Inquiry Examples
        "Collaboration Proposal for AI Research", 
        "Request for Access to Heart Disease Diagnostic Data", 
        "Joint Grant Proposal for AI in Autonomous Driving", 
        "Proposal for Co-authoring a Paper on 'Explainable AI'", 
        "Invitation to Present Research on Climate Prediction", 
        "Request for TensorFlow Code for Autonomous Vehicles", 
        "Invitation to Lead a Workshop on Deep Learning", 
        "Data Sharing Agreement for Autonomous Vehicle Data", 
        "Request for Help with IRB Submission", 
        "Proposal to Co-author a Review Paper on AI",

        # Corporate Inquiry Examples
        "AI Internship Opportunities for Students", 
        "Proposal for Partnership on AI-Driven Healthcare", 
        "Sponsorship Request for Autonomous Driving Research", 
        "Looking to Recruit Graduates for AI Roles", 
        "Request for AI Training Program for Employees", 
        "Inquiry About Licensing Your AI Diagnostic Model", 
        "Request for Consulting on AI Model Development", 
        "Confidential Terms Discussion for AI Research", 
        "Invitation to Provide a Guest Lecture on AI", 
        "Request for Feedback on Our AI Healthcare Product"
    ],
    "example_body": [
        # Student Inquiry Example Bodies
        "I would like to request the syllabus for the course 'Introduction to AI'. Could you please share the weekly topics, reading materials, and assignment deadlines?", 
        "Could you please let me know the due date for the assignment on Neural Networks in 'Advanced Machine Learning'?", 
        "When is the midterm exam for 'Natural Language Processing'? Could you also provide details on the format?", 
        "Could you tell me when and where Professor Jane Doe holds office hours for 'Introduction to AI'?", 
        "Are there any tutoring sessions available for 'Advanced Machine Learning'? I would like to attend.", 
        "Has the lab session for 'Introduction to AI' been rescheduled due to the holiday next week?", 
        "How can I appeal my grade for the final exam in 'Natural Language Processing'?", 
        "How many credits do I need to graduate with a major in Computer Science?", 
        "I would like to know if I can join the 'AI for Healthcare Diagnostics' research project under your supervision.", 
        "Could you please share the reading materials for the first week of 'Advanced Machine Learning'?",

        # Academic Collaboration Inquiry Example Bodies
        "I would like to propose a collaboration on your 'AI for Healthcare Diagnostics' project. Could we discuss potential areas where my team could contribute?", 
        "Could you please provide access to the heart disease diagnostic dataset for your healthcare research? I am interested in conducting some comparative studies.", 
        "I am writing to inquire if you'd be interested in submitting a joint research grant proposal on autonomous driving AI.", 
        "I would like to propose that we co-author a research paper on 'Explainable AI' and its applications in healthcare.", 
        "We would like to invite you to present your research on climate change prediction models at our upcoming AI symposium.", 
        "Could you share the TensorFlow code used for your reinforcement learning model for autonomous vehicle navigation?", 
        "We would like to invite you to lead a workshop on deep learning architectures at our upcoming research seminar.", 
        "Can we arrange a data-sharing agreement for the autonomous vehicle data from your recent research?", 
        "Could you assist us with the preparation of the IRB submission for our collaborative AI healthcare research?", 
        "Would you be interested in co-authoring a review paper on AI applications in climate simulations?",

        # Corporate Inquiry Example Bodies
        "We are looking for AI interns for summer positions in our development team. Could we discuss opportunities for collaboration with your students?", 
        "We would like to propose a strategic partnership with your university to develop AI-driven healthcare solutions.", 
        "We are interested in sponsoring your ongoing research in autonomous vehicle navigation. Could we discuss further?", 
        "Our company is currently hiring for AI and machine learning roles. We would like to recruit some of your recent graduates.", 
        "Could you offer a custom AI training program for our employees focused on machine learning applications?", 
        "We are interested in licensing the AI diagnostic model you developed. Could you provide more details?", 
        "Would you be available to consult on an AI model for predictive maintenance in autonomous systems?", 
        "We would like to discuss the confidential terms of our AI research partnership on healthcare diagnostics.", 
        "We would like to invite you to deliver a guest lecture at our upcoming corporate AI conference.", 
        "Could you provide feedback on the AI product we’ve developed for healthcare diagnostics? We would value your input."
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

def generate_prompt(email_class, category, example_subject, example_body):
    prompt = f"""
    You are a powerful language model tasked with generating realistic emails for various categories. The emails will be used to simulate common inquiries received by a university Head of Department (HOD). These emails fall into three classes: "Student Inquiry," "Academic Collaboration Inquiry," and "Corporate Inquiry." You need to generate emails for the class "{email_class}" in the category "{category}".
    
    For this class and category, generate 10 emails. Each email must have a:
    1. Sender Email Address: A realistic email address matching the type of inquiry (e.g., a student for student inquiries, a researcher for academic collaboration, or a corporate entity for corporate inquiries).
    2. Subject Line: A brief subject line that clearly reflects the email's purpose.
    3. Email Body: A 2-3 paragraph email body that sounds natural and relates to the category.

    Here's an example email structure for the category "{category}":
    
    Sender: student123@university.edu
    Subject: {example_subject}
    
    Dear Professor,

    {example_body}

    Best regards,
    [Student's Name]
    
    Make sure the emails you generate are realistic, follow the example structure, and provide proper details based on the category. The output should be formatted in CSV with the following columns: sender, subject, body, class.
    
    Generate at least 30 emails for this category.
    """
    return prompt

# Example usage for the "Request for Course Syllabus" category under "Student Inquiry"

# Initialize a list to store the generated prompts
prompts = []

# Loop over each row in the DataFrame
for index, row in df.iterrows():
    prompt = generate_prompt(
        email_class=row['email_class'],
        category=row['category'],
        example_subject=row['example_subject'],
        example_body=row['example_body']
    )
    prompts.append(prompt)

prompt_df = pd.DataFrame(prompts, columns=['prompt'])
prompt_df.to_csv('generated_prompts.csv', index=False)
print(prompt_df)
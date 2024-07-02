from flask import Flask, request, jsonify, send_file
from utilities import fetch_data
from dotenv import dotenv_values
from groq import Groq
from tqdm import tqdm
from sklearn.cluster import MeanShift
import numpy
import ollama
import time
from typing import Literal
from markdown_pdf import MarkdownPdf, Section
import os

app = Flask(__name__)

variables = dotenv_values(".env")
gen_model = "llama3-8b-8192"
embedding_model = "nomic-embed-text"
#-------------
_INPUTS = Literal["text","url","youtube","pdf"]
temp_path = "Temp/"

os.makedirs(temp_path, exist_ok=True)
out_path = temp_path +"temp.pdf"
client = Groq(api_key=variables["GROQ_API_KEY"])


#-------------------------------
topic_prompt = '''
Given the following excrepts compiled from textbooks and lecture transcripts on a subject.

{content}

Identify core topics discussed and provide them an importance score.
'''

content_prompt = '''
Given the following excrepts compiled from textbooks and lecture transcripts on a subject.

{content}

Clean the contents and make a comprehensive lecture notes on the topics being covered. Stick to the contents
'''

question_prompt = ''' 
Given the following lecture notes.

<lecture_notes>
{lecture_notes}
</lecture_notes>

Topic importance of each topic discussed in the lecture is given below.

<topic importance>
{topic_imp}
<topic importance>
You are a Teacher tasked with setting up a large number of questions for an upcoming examination. The number of questions per topic should depend upon the topic importance.
The questions should include conceptual, reasoning and application level questions. Do not generate answers. Generate questions and not a question distribution
'''

#------------------------

def get_embedding(text):
   '''
   Python function that takes in a text input and generates a embeddng based on mistral model.
   
   '''
   return ollama.embeddings(model=embedding_model, prompt=text)['embedding']

def syllabus(content_clusters):
    '''
    Model that takes in information from each content clusters and generates lecture notes and topic importance between each cluster.
    '''

    labels = set(content_clusters.values())
    syllabus_list = []
    for label in tqdm(labels):
        content = "\n".join([ct for ct, lb in content_clusters.items() if lb == label])

        topic_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful professor tasked with teaching and testing knowledge of students."
                },
                {
                    "role": "user",
                    "content": topic_prompt.replace("{content}", content),
                }
            ],
            model=gen_model
        ).choices[0].message.content

        content_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful professor tasked with teaching and testing knowledge of students."
                },
                {
                    "role": "user",
                    "content": content_prompt.replace("{content}", content),
                }
            ],
            model=gen_model
        ).choices[0].message.content

        syllabus_list.append((topic_response, content_response))
        time.sleep(7)

    return syllabus_list


def generate_questions(content_tuple):
    '''
    Function that takes in the lecture notes and topic importance generated in the last step and generates question papers based on various aspects of the problem.
    '''
    content_dict = {"Lecture Note": content_tuple[1], "Topic Importance": content_tuple[0]}
    question_content = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful professor tasked with teaching and testing knowledge of students."
            },
            {
                "role": "user",
                "content": question_prompt.replace("{lecture_notes}", content_tuple[1]).replace("{topic_imp}", content_tuple[0]),
            }
        ],
        model=gen_model
    ).choices[0].message.content
    content_dict["Question Paper"] = question_content
    time.sleep(30)
    return content_dict



def write_chapters(content_dict:dict, out_path:str) -> str:
    out_pdf = MarkdownPdf()
    out_content = ""
    out_pdf.add_section(Section("# Content Summary\n"))
    for cd in content_dict:
        out_content += "## Section\n"
        out_content += "### Topics Discussed\n"
        out_content += cd["Topic Importance"] + "\n"
        out_content += "### Notes\n"
        out_content += cd["Lecture Note"] + "\n"
        out_content += "### Sample Questions\n"
        out_content += cd["Question Paper"] + "\n\n"
    out_pdf.add_section(Section(out_content))
    out_pdf.save(out_path)


@app.route('/generate_content', methods=['POST','GET'])
def generate_all_content():
    content_type = request.form['content_type']
    input_images_pdf = request.form.get('input_images_pdf', False)

    if content_type.lower() not in ["text", "url", "pdf"]:
        return jsonify({"error": "Unsupported input: Currently the following data sources are supported. text, url, youtube, pdf"}), 400
    elif content_type.lower() == "pdf":
        file = request.files['file']
        content = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(content)
    else:
        content = request.form['content']

    content_text = fetch_data.fetch_input(content, content_type,input_images_pdf)
    content_text = [ct for ct in tqdm(content_text) if ct.replace("\n","").replace(" ","") != ""]
    content_embedding = [get_embedding(ct) for ct in tqdm(content_text)]
    content_embedding = numpy.array(content_embedding)
    clusters = MeanShift().fit(content_embedding)
    content_clusters = dict(zip(content_text, clusters.labels_))
    content_model = syllabus(content_clusters)
    content_dict = [generate_questions(ct) for ct in content_model]
    write_chapters(content_dict, out_path)
    return send_file(out_path, as_attachment=True)

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
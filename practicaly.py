from utilities import fetch_data
from dotenv import dotenv_values
import openai 
from tqdm import tqdm
from utilities import fetch_data
from sklearn.cluster import MeanShift
import numpy
from markdown_pdf import MarkdownPdf,Section
import mistralai.client
from typing import Literal


#-------------
_INPUTS = Literal["text","url","youtube","pdf"]

variables = dotenv_values(".env")
gen_model = "mistral-large-latest"
embedding_model = "mistral-embed"
openai.api_key = variables["OPENAI_API_KEY"] #For video support
content = variables["CONTENT"]
content_type = variables["CONTENT_TYPE"]
client = mistralai.client.MistralClient(api_key=variables["MISTRAL_API_KEY"])

out_path = variables["OUTPUT_PATH"]
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
   return client.embeddings(model=embedding_model,input=[text]).data[0].embedding

def syllabus(content_clusters):
  '''
  Model that takes in information from each content clusters and generates lecture notes and topic importance between each cluster.
  
  '''
  
  labels = set(content_clusters.values())
  syllabus_list = []
  for label in tqdm(labels):
    content = "\n".join([ct for ct,lb in content_clusters.items() if lb == label])
    topic_response = client.chat(model=gen_model,messages=[mistralai.models.chat_completion.ChatMessage(role="user", content=topic_prompt.replace("{content}",content))]).choices[0].message.content
    content_response = client.chat(model=gen_model,messages=[mistralai.models.chat_completion.ChatMessage(role="user", content=content_prompt.replace("{content}",content))]).choices[0].message.content
    syllabus_list.append((topic_response,content_response))
  return syllabus_list

def generate_questions(content_tuple):
    '''
    Function that takes in the lecture notes and topic importance generated in the last step and generates question papers based on various aspects of the problem.

    
    '''
    content_dict ={"Lecture Note":content_tuple[1],"Topic Importance": content_tuple[0]}
    question_content = client.chat(model=gen_model,messages=[mistralai.models.chat_completion.ChatMessage(role="user", content=question_prompt.replace("{lecture_notes}",content_tuple[1]).replace("{topic_imp}",content_tuple[0]))]).choices[0].message.content
    content_dict["Question Paper"] = question_content
    return content_dict



def write_chapters(content_dict:dict, out_path:str) -> str:
    '''
    Function that takes a list of dictionary containing the various information on the content and writes the output as a PDF file.
    
    '''
    out_pdf = MarkdownPdf()
    out_content = ""  # Initialize out_content variable
    out_pdf.add_section(Section("# " + content.split("/")[-1].split(".")[0] + "\n"))
    for cd in content_dict:
        out_content += "## Section 01\n"
        out_content += "### Topics Discussed\n"
        out_content += cd["Topic Importance"] + "\n"
        out_content += "### Notes\n"
        out_content += cd["Lecture Note"] + "\n"
        out_content += "### Sample Questions\n"
        out_content += cd["Question Paper"] + "\n\n"
    out_pdf.add_section(Section(out_content))
    out_pdf.save(out_path)


def generate_all_content(content:str,content_type:_INPUTS="text",input_images_pdf:bool=False):
    '''
    Function that takes in a content and content type and generates Lectures Notes, Topic Importance and Question Papers based on the content.
        - content: The content input and the content_type.
            content is path for urls , pdf or youtube . For text , it is the text itself.
        - content_type: The type of content. It can have the following values  text, url, youtube, pdf
        - input_images_pdf: Set to true when dealing with image heavy PDFs. Would result in heavy computation times. Uses llava model via Ollama
    '''
    if content_type.lower() not in _INPUTS.__args__:
        raise ValueError("Unsupported input: Currently the following data sources are supported. text,url,youtube,pdf")
    else:
        content_text = fetch_data.fetch_input(content,content_type,input_images_pdf)
        content_text = [ct for ct in tqdm(content_text) if ct.replace("\n","").replace(" ","") != ""]
        content_embedding = [get_embedding(ct) for ct in tqdm(content_text)]
        content_embedding = numpy.array(content_embedding)
        clusters = MeanShift().fit(content_embedding)
        content_clusters = dict(zip(content_text,clusters.labels_))
        content_model = syllabus(content_clusters)
        content_dict = [generate_questions(ct) for ct in content_model]
        write_chapters(content_dict,out_path)


generate_all_content(content,content_type)
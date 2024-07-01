from typing import Literal
from bs4 import BeautifulSoup
from bs4.element import Comment,NavigableString
from urllib.request import Request, urlopen
from pytube import YouTube
from pypdf import PdfReader
import uuid
import os
import openai
import ollama

import warnings
from dotenv import dotenv_values

api_keys = dotenv_values(".env")
openai.api_key = api_keys["OPENAI_API_KEY"]

audio_client = openai.OpenAI(api_key=openai.api_key)
_INPUTS = Literal["text","url","youtube","pdf"]

import ollama
import os
from tqdm import tqdm

from  pypdf import PdfReader

image_prompt = '''
    Provide a descriptive summary of the diagrams provided. The diagram are extracted from the document {name}. Image:
'''

def fetch_pdf(path:str,support_image:bool=True,base_path:str="Data/") -> str:
    '''Function that takes in a PDF file as returns content present in it.'''
    reader = PdfReader(path)
    content = ""
    number_of_pages = len(reader.pages)
    temp_path = base_path + "temp.png"
    for pg in range(number_of_pages):
        content = content + reader.pages[pg].extract_text()
        content = content + "\n\n"
        image_files = reader.pages[pg].images
        pdf_name = path.split("/")[-1]
        if len(image_files) > 0 and support_image:
            if pg == 0:
                continue
            else:
                try:
                    for image_file_object in image_files:
                        with open(temp_path, "wb") as fp:
                            fp.write(image_file_object.data)
                        response = ollama.chat(
                        model="llava",
                        messages=[
                        {
                        'role': 'user',
                        'content': image_prompt.replace("{name}",pdf_name),
                        'images': [temp_path]
                        }
                        ]
                    )
                        content = content +"\n" + response["message"]["content"]
                except:
                    continue
    return content

def tag_visible(element):
    '''
    Function to identify visible text elements from a given webpage.
    These are text data part of the following parent tags.

    'style', 'script', 'head', 'title', 'meta', '[document]','i'

    Comments and Navigable Strings are also excluded. So are headings belonging to elements in 'dropdown-title' class
    '''
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]','i']:
        return False
    if 'class' in element.attrs:
        if "dropdown-title" in element.attrs['class']:
            return False
        if "btn" in element.attrs["class"]:
            return False
        if "nav__title" in element.attrs["class"]:
            return False
            return False
    if isinstance(element, Comment):
        return False
    if isinstance(element,NavigableString):
        return False
    return True


def text_from_html(body):
    '''
    Extracting text from webpage
    '''
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

def extract_text_from_website(url:str) -> str:
    '''
    Function to extract text content from a given website. Takes the url as an input.

    It uses Mozilla User Agent to access webpages and returns texts that are part of 'h1', 'h2', 'h3', 'h4', 'h5', 'h6','p' tags.
    Filters out unwanted html content.
    '''
    try:
        req = Request(url,headers={'User-Agent':"Mozilla/5.0"})
        html = urlopen(req)
        soup = BeautifulSoup(html,'html.parser')
        text = soup.findAll(['h1', 'h2', 'h3', 'h4', 'h5', 'h6','p'])
        visible_texts = filter(tag_visible, text)
        return "\n".join([txt.text for txt in visible_texts])
    except:
        raise Exception("Something went wrong with the URL")

def download_audio(url:str,base_path:str="Data/") -> str:
  '''
  Function that takes in a youtube url and downloads the file to a given path. The function returns the path of the downloaded file

  '''
  file_path = base_path + str(uuid.uuid1()) + ".mp3"
  video = YouTube(url)
  audio = video.streams.filter(only_audio=True).first()
  audio_out = audio.download(base_path)

  # Renaming the file
  _, ext = os.path.splitext(audio_out)
  os.rename(audio_out,file_path)

  return file_path


def youtube_to_transcript(url:str,del_audio:bool = True,base_path:str="Data/") -> str:
  '''
  Function that takes in an url and converts it into transcripts. It takes two variables.
    - url : The youtube URL for the video.
    - base_path: Path where the audios are stored for transcription. The audio is deleted after transcription.
    - del_audio: Wether audio needs to be deleted post transcription. Used for testing purpouses. Should be set to True in production.

    The function currently supports only English videos. But can be later used to translate and transcribe other openai supported languages.
    Additionally, the audio size is limited to 25MB as of now
  '''

  audio_path = download_audio(url,base_path)
  audio_file = open(audio_path,'rb')
  transcript = audio_client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
  if del_audio:
    os.remove(audio_path)

  return transcript.text



def fetch_input(content:str,type: _INPUTS= "text",input_images_pdf:bool=False) -> str:
    '''
    Returns text from a given input. The input could be either text or url.
    If URL the text contents from the website is fetched. Text is passed as is.

    Example Usage:
            >>> data_ingest.fetch_input("Hello World")
                'Hello World'

            >>> data_ingest.fetch_input("Hello World","text")
                'Hello World'

            >>> data_ingest.fetch_input("https://sanjaii.github.io/","url")
                'Sanjay Karukamanna\n\n    My name is Sanjay and I live in Kerala. Currently working as Software Engineer. Mostly engaged with\n    Ruby, Go, Bash and Occasionally JavaScript.\n\n\n\n    Blog Posts\n\n\n    Elsewhere\n\n\n        © 2021\n        hi@sanjay.link\n        |\n        \n        |\n        \n        |\n        \n        |\n        \n        |\n        \n\n\n        Hosted on a Raspberry Pi with the help of frp\n'


    '''
    if type.lower() not in _INPUTS.__args__:
        raise ValueError("Unsupported input")
    else:
        try:
            if type == "text":
                return content.split("\n\n")
            if type == "url":
                return extract_text_from_website(content).split("\n\n")
            if type == "youtube":
                return youtube_to_transcript(content).split("\n\n")
            if type == "pdf":
                return fetch_pdf(content,support_image=input_images_pdf).split("\n\n")
        except:
            warnings.warn("Something went wrong, ignoring current file")


    


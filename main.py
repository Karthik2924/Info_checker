import pathlib
import textwrap
import gradio as gr
import bs4 as bs
import urllib.request
import gradio as gr
from transformers import pipeline
import numpy as np
from googlesearch import search
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

with open('tokens.json', 'r') as openfile:
	json_object = json.load(openfile)

access_token = json_object['hf']
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",token = access_token)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b-it",
    torch_dtype=torch.bfloat16,
    token=access_token
).to('cuda')

def llm_inference(txt):
  input_ids = tokenizer(txt, return_tensors="pt").to("cuda")
  l = len(input_ids['input_ids'][0])
  # model.to('cuda')
  outputs = model.generate(**input_ids,max_new_tokens = 1000)#,do_sample= True,temperature = 0.5)
  return tokenizer.decode(outputs[0][l:])

#transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]

def llm_output(txt,query):
    prompt= " The above is an information source, does the main idea of the source verify the following query. answer in around 20 words "
    # prompt2 = "List the claims in the following text "
    return llm_inference(txt + prompt + query)
    # response = model.generate_content(txt + prompt + query)
    # return response.candidates[0].content.parts[0].text

def process(text,num_searchs=3):
  url_list = get_urls( text,num_searchs)
  processed = []
  unp = []
  for i in  url_list:
      try:
          processed.append(process_url(i))
      except:
          unp.append(i)
  llm_outs = []
  for i in processed:
    llm_outs.append(llm_output(i,text))#+'\n')
  return llm_outs,url_list

def main(text, num_searchs,audio,video):
    if text != "":
      ans,sources = process(text,num_searchs)
    elif audio!= None:
      txt = transcribe(audio)
      ans,sources = process(text,num_searchs)
    elif video!=None:
      txt = transcribe(video)
      ans,sources = process(text,num_searchs)
    else:
      ans = "please submit an input"
      return ans
    
    final_ans = ""
    for i in ans:
      final_ans+=i
      final_ans+='\n'
    final_ans+= "The following are the sources" + '\n'
    for i in sources:
      final_ans+=i
      final_ans+='\n'

    return final_ans

def get_urls(query,num_results):
    return list(search(term = query,num_results=num_results))

def process_url(url):
    source = urllib.request.urlopen(url)
    soup = bs.BeautifulSoup(source,'lxml')
    txt = soup.text.replace('\n','').replace('\t','')
    return txt


def llm_define_input(txt):
    prompt1 = "Give a one line explanation of the main idea of the following content "
    # prompt2 = "List the claims in the following text "
    return llm_inference(prompt1 + txt)
    response = model.generate_content(prompt1 + txt)
    return response.candidates[0].content.parts[0].text
    
def llm_output(txt,query):
    prompt= " The above is an information source, does the main idea of the source verify the following query "
    # prompt2 = "List the claims in the following text "
    return llm_inference(txt + prompt + query)
    response = model.generate_content(txt + prompt + query)
    return response.candidates[0].content.parts[0].text


demo = gr.Interface(
    fn=main,
    inputs=["text", gr.Slider(value=2, minimum=1, maximum=10, step=1),"audio","video"],
    outputs=[gr.Textbox(label="output", lines=10)],
)

if __name__ == "__main__":
    demo.launch(debug = True)

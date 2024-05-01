# Info_checker
Verify information from different sources with completely open source models.
* Gemma 2B model with float16 from huggingface.
* Google Search api with python and beautiful soup 4 for web scracping and sources extraction.
* Pytorch backend.
* Gradio for hosting the web UI.

### Future updates:
* Connect to a db and store queries and responses
* Reduce latency
  * try smaller but reliable models?
  * parallelize scraping sources and summarizing or try batched inference on multiple sources with limited compute.
* Increase relavance and reliability by scraping specific websites which are not blocked.


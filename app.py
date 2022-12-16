import gradio as gr
import numpy as np
from transformers import pipeline

pipe = pipeline("sentiment-analysis", ,model="siebert/sentiment-roberta-large-english")

def message_sentiment(text):
  return pipe(text)[0]

def batch_sentiment(texts):
    return pipe(texts)

def parse_into_paragraphs(single_string):
    return list(filter(bool, single_string.splitlines()))

def format_list(strings):
    return '\n'.join(strings)

def analyse_batch_sentiment(text):
    texts = parse_into_paragraphs(text)
    sentiments = batch_sentiment(texts)
    return_values = []
    for index in range(len(texts)):
        return_values.append(texts[index] + ": " + str(sentiments[index]))
    return format_list(return_values)

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with gr.Blocks() as demo:
    gr.Markdown("Use one of the tabs to switch between different tasks")
    with gr.Tab("Single paragraph sentiment"):
        s_sentiment_input = gr.Textbox()
        s_sentiment_output = gr.Textbox()
        s_sentiment_button = gr.Button("Analyze sentiment")
    with gr.Tab("Batch paragraph sentiment raw"):
        gr.Markdown("Enter list of paragraphs in the first field. System will break it into paragraphs, and calculate sentiment scores for each of them.")
        b_sentiment_input = gr.Textbox(lines=5)
        b_sentiment_output = gr.Textbox(lines=5)
        b_sentiment_button = gr.Button("Analyze sentiment")
#     with gr.Tab("Flip Image"):
#         with gr.Row():
#             image_input = gr.Image()
#             image_output = gr.Image()
#         image_button = gr.Button("Flip")

#     with gr.Accordion("Open for More!"):
#         gr.Markdown("Look at me...")

    s_sentiment_button.click(message_sentiment, inputs=s_sentiment_input, outputs=s_sentiment_output)
    b_sentiment_button.click(analyse_batch_sentiment, inputs=b_sentiment_input, outputs=b_sentiment_output)
#     image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()
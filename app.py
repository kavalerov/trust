import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

sentiment_pipe = pipeline(
    "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
)

embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def message_sentiment(text):
    return sentiment_pipe(text)[0]


def batch_sentiment(texts):
    return sentiment_pipe(texts)


def parse_into_paragraphs(single_string):
    return list(filter(bool, single_string.splitlines()))


def format_list(strings):
    return "\n".join(strings)


def analyse_batch_sentiment(text):
    texts = parse_into_paragraphs(text)
    sentiments = batch_sentiment(texts)
    return_values = []
    for index in range(len(texts)):
        return_values.append([texts[index], str(sentiments[index])])
    return return_values


def compare_semantic_similarity(set_of_sentences, target_sentence):
    texts = parse_into_paragraphs(set_of_sentences)
    embeddings = embeddings_model.encode(texts, convert_to_tensor=True)
    target_embedding = embeddings_model.encode(target_sentence, convert_to_tensor=True)
    return_values = []
    for index in range(len(texts)):
        similarity = util.pytorch_cos_sim(embeddings[index], target_embedding)[0][
            0
        ].item()

        return_values.append([texts[index], similarity])
    return return_values


with gr.Blocks() as demo:
    gr.Markdown("Use one of the tabs to switch between different tasks")
    with gr.Tab("Single paragraph sentiment"):
        s_sentiment_input = gr.Textbox()
        s_sentiment_output = gr.Textbox()
        s_sentiment_button = gr.Button("Analyze sentiment")
        s_sentiment_button.click(
            message_sentiment, inputs=s_sentiment_input, outputs=s_sentiment_output
        )
    with gr.Tab("Batch paragraph sentiment raw"):
        gr.Markdown(
            "Enter list of paragraphs in the first field. System will break it into paragraphs,\
                 and calculate sentiment scores for each of them."
        )
        b_sentiment_input = gr.Textbox(lines=5)
        b_sentiment_output = gr.JSON()
        b_sentiment_button = gr.Button("Analyze sentiment")
        b_sentiment_button.click(
            analyse_batch_sentiment,
            inputs=b_sentiment_input,
            outputs=b_sentiment_output,
        )
    with gr.Tab("Compare semantic similarity of one sentence to a set of sentences"):
        gr.Markdown(
            "Enter list of paragraphs in the first field. System will break it into paragraphs, \
                and calculate semantic similarity of sentence in the second field to each of the sentences in the first paragraph."
        )
        set_similarity_input_set = gr.Textbox(lines=5)
        set_similarity_target = gr.Textbox(lines=5)
        set_similarity_button = gr.Button("Compare semantic similarity")
        set_similarity_output = gr.DataFrame(wrap=True)
        set_similarity_button.click(
            compare_semantic_similarity,
            inputs=[set_similarity_input_set, set_similarity_target],
            outputs=set_similarity_output,
        )

    with gr.Tab("Average semantic similarity of one sentence to a set of sentences"):
        gr.Markdown(
            "Enter list of paragraphs in the first field. System will break it into paragraphs, \
                and calculate average semantic similarity of sentence in the second field to each of the sentences in the first paragraph."
        )
        # b_sentiment_input = gr.Textbox(lines=5)
        # b_sentiment_output = gr.Textbox(lines=5)
        # b_sentiment_button = gr.Button("Compare semantic similarity")
    with gr.Tab("Sentiment timeline visualisation"):
        gr.Markdown(
            "Enter list of paragraphs in the first field. System will break it into paragraphs, \
                and calculate sentiment scores for each of them, and then product a visalisation of how the sentiment has changed over time."
        )
        # b_sentiment_input = gr.Textbox(lines=5)
        # b_sentiment_output = gr.Textbox(lines=5)
        # b_sentiment_button = gr.Button("Analyze sentiment")
    with gr.Tab('Use of "we" words'):
        gr.Markdown(
            'Enter list of paragraphs in the first field. System will break it into paragraphs, \
                and identify the ones that use "we" words, and then product a visalisation of how the sentiment has changed over time.'
        )
        # b_sentiment_input = gr.Textbox(lines=5)
        # b_sentiment_output = gr.Textbox(lines=5)
        # b_sentiment_button = gr.Button("Analyze sentiment")
    #     with gr.Tab("Flip Image"):
    #         with gr.Row():
    #             image_input = gr.Image()
    #             image_output = gr.Image()
    #         image_button = gr.Button("Flip")

    #     with gr.Accordion("Open for More!"):
    #         gr.Markdown("Look at me...")


#     image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()

import re

import gradio as gr
import numpy as np
import pandas as pd
from plotly import express as px

# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline

# sentiment_pipe = pipeline(
#     "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
# )

# embeddings_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# def message_sentiment(text):
#     return sentiment_pipe(text)[0]


# def batch_sentiment(texts):
#     return sentiment_pipe(texts)


# def parse_into_paragraphs(single_string):
#     return list(filter(bool, single_string.splitlines()))


# def format_list(strings):
#     return "\n".join(strings)


# def analyse_batch_sentiment(text):
#     texts = parse_into_paragraphs(text)
#     sentiments = batch_sentiment(texts)
#     return_values = []
#     for index in range(len(texts)):
#         return_values.append([texts[index], str(sentiments[index])])
#     return return_values


# def compare_semantic_similarity(set_of_sentences, target_sentence):
#     texts = parse_into_paragraphs(set_of_sentences)
#     embeddings = embeddings_model.encode(texts, convert_to_tensor=True)
#     target_embedding = embeddings_model.encode(target_sentence, convert_to_tensor=True)
#     return_values = []
#     for index in range(len(texts)):
#         similarity = util.pytorch_cos_sim(embeddings[index], target_embedding)[0][
#             0
#         ].item()

#         return_values.append([texts[index], similarity])
#     return return_values


# def visualise_sentiment_timeline(set_of_sentences):
#     texts = parse_into_paragraphs(set_of_sentences)
#     sentiments = batch_sentiment(texts)
#     scores = [
#         float(value["score"])
#         if float(value["score"] >= 0)
#         else -(float(value["score"]))
#         for value in sentiments
#     ]
#     colors = [
#         "red" if value["score"] == "NEGATIVE" else "green" for value in sentiments
#     ]
#     fig, ax = plt.subplots()
#     ax.bar(range(len(sentiments)), scores, color=colors)
#     return fig


# with gr.Blocks() as demo:
#     gr.Markdown("Use one of the tabs to switch between different tasks")
#     with gr.Tab("Single paragraph sentiment"):
#         s_sentiment_input = gr.Textbox()
#         s_sentiment_output = gr.Textbox()
#         s_sentiment_button = gr.Button("Analyze sentiment")
#         s_sentiment_button.click(
#             message_sentiment, inputs=s_sentiment_input, outputs=s_sentiment_output
#         )
#     with gr.Tab("Batch paragraph sentiment raw"):
#         gr.Markdown(
#             "Enter list of paragraphs in the first field. System will break it into paragraphs,\
#                  and calculate sentiment scores for each of them."
#         )
#         b_sentiment_input = gr.Textbox(lines=5)
#         b_sentiment_output = gr.JSON()
#         b_sentiment_button = gr.Button("Analyze sentiment")
#         b_sentiment_button.click(
#             analyse_batch_sentiment,
#             inputs=b_sentiment_input,
#             outputs=b_sentiment_output,
#         )
#     with gr.Tab("Compare semantic similarity of one sentence to a set of sentences"):
#         gr.Markdown(
#             "Enter list of paragraphs in the first field. System will break it into paragraphs, \
#                 and calculate semantic similarity of sentence in the second field to each of the sentences in the first paragraph."
#         )
#         set_similarity_input_set = gr.Textbox(lines=5)
#         set_similarity_target = gr.Textbox(lines=5)
#         set_similarity_button = gr.Button("Compare semantic similarity")
#         set_similarity_output = gr.DataFrame(wrap=True)
#         set_similarity_button.click(
#             compare_semantic_similarity,
#             inputs=[set_similarity_input_set, set_similarity_target],
#             outputs=set_similarity_output,
#         )

#     with gr.Tab("Average semantic similarity of one sentence to a set of sentences"):
#         gr.Markdown(
#             "Enter list of paragraphs in the first field. System will break it into paragraphs, \
#                 and calculate average semantic similarity of sentence in the second field to each of the sentences in the first paragraph."
#         )
#         # b_sentiment_input = gr.Textbox(lines=5)
#         # b_sentiment_output = gr.Textbox(lines=5)
#         # b_sentiment_button = gr.Button("Compare semantic similarity")
#     with gr.Tab("Sentiment timeline visualisation"):
#         gr.Markdown(
#             "Enter list of paragraphs in the first field. System will break it into paragraphs, \
#                 and calculate sentiment scores for each of them, and then product a visalisation of how the sentiment has changed over time."
#         )
#         timeline_input = gr.Textbox(lines=5)
#         timeline_button = gr.Button("Visualize sentiment")
#         timeline_output = gr.Plot()
#         timeline_button.click(
#             visualise_sentiment_timeline,
#             inputs=timeline_input,
#             outputs=timeline_output,
#         )
#     with gr.Tab('Use of "we" words'):
#         gr.Markdown(
#             'Enter list of paragraphs in the first field. System will break it into paragraphs, \
#                 and identify the ones that use "we" words, and then product a visalisation of how the sentiment has changed over time.'
#         )
#         # b_sentiment_input = gr.Textbox(lines=5)
#         # b_sentiment_output = gr.Textbox(lines=5)
#         # b_sentiment_button = gr.Button("Analyze sentiment")
#     #     with gr.Tab("Flip Image"):
#     #         with gr.Row():
#     #             image_input = gr.Image()
#     #             image_output = gr.Image()
#     #         image_button = gr.Button("Flip")

#     #     with gr.Accordion("Open for More!"):
#     #         gr.Markdown("Look at me...")


# #     image_button.click(flip_image, inputs=image_input, outputs=image_output)


def process(history, num_people, we_dict):
    print(history)
    regex = r"^([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d),\s([0-1]?[0-9]|2?[0-3]):([0-5]\d) - (.*):"
    intervention_datetimes = re.findall(regex, history, re.MULTILINE)
    dates = []
    active_participants = []
    df = pd.DataFrame(columns=["total"])
    print(intervention_datetimes)
    for item in intervention_datetimes:
        participant_name = item[7]
        if participant_name not in active_participants:
            active_participants.append(participant_name)
            print(participant_name)
            df[participant_name] = [] if len(dates) == 0 else [0] * len(dates)
        day = (
            item[0] if len(item[0]) > 0 else (item[1] if len(item[1]) > 0 else item[2])
        )
        full_date = day + "/" + item[3] + "/" + item[4]
        if full_date not in dates:
            df.loc[full_date] = [0 for i in range(len(active_participants) + 1)]
            dates.append(full_date)
        df[participant_name].loc[[full_date]] += 1
        df["total"].loc[full_date] += 1

    # formatted_dates = []
    # for key, value in dates.items():
    #     print(key, value)
    #     formatted_dates.append([key, value])
    # print(formatted_dates)
    num_of_questions = history.count("?")
    bar_plot = px.bar(df, x=df.index, y="total")

    return df, bar_plot, "", str(num_of_questions), ""


with gr.Blocks() as demo:
    gr.Label("Chat history analysis")
    chat_history = gr.Textbox(
        "Enter chat history here",
        interactive=True,
        max_lines=500,
        lines=10,
        label="Chat history",
    )
    num_of_people = gr.Textbox(
        "Enter number of people in the chat here",
        interactive=True,
        max_lines=1,
        lines=1,
        label="Total number of people in the chat",
    )
    we_dictionary = gr.Textbox(
        "Enter dictionary of 'we' words here",
        interactive=True,
        max_lines=500,
        lines=10,
        label="Dictionary of words to check for occurance",
    )
    submit_btn = gr.Button(value="Submit")
    gr.Label("Output")
    # output_interventions = gr.Textbox(
    #     "Interventions will be displayed here",
    #     max_lines=500,
    #     lines=10,
    #     label="Interventions",
    # )
    output_interventions = gr.DataFrame(headers=["Date", "Number of interventions"])
    output_interventions_plot = gr.Plot()
    output_proportion = gr.Textbox(
        "Proportion of 'we' words will be displayed here",
        max_lines=500,
        lines=10,
        label="Proportion of active usage",
    )
    output_we_words = gr.Textbox(
        "Usage of 'we' words will be displayed here",
        max_lines=500,
        lines=10,
        label="Usage of preset words",
    )
    output_num_of_question_marks = gr.Textbox(
        "Question marks usage will be displayed here",
        max_lines=500,
        lines=10,
        label="Number of question marks",
    )
    submit_btn.click(
        process,
        inputs=[chat_history, num_of_people, we_dictionary],
        outputs=[
            output_interventions,
            output_interventions_plot,
            output_proportion,
            output_we_words,
            output_num_of_question_marks,
        ],
    )

    gr.Label("Output")


demo.launch()

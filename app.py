import re

import gradio as gr
import numpy as np
import pandas as pd
from plotly import express as px


# function that takes in a string and returns a list of words
def process_text(text):
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # remove whitespace
    text = text.strip()
    # convert to lowercase
    text = text.lower()
    # split into words
    words = text.split(" ")
    return words


# Function to calculate the number of words in a string
def count_words(text):
    words = process_text(text)
    return len(words)


# Function creates a histogram of the words in a string
def create_histogram(text):
    words = process_text(text)
    word_counts = {}
    for word in words:
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1
    return word_counts


# Function that draws a histogram of the words in a string
def draw_histogram(text):
    word_counts = create_histogram(text)
    return px.bar(x=list(word_counts.keys()), y=list(word_counts.values()))


def process(history, num_people, we_dict):
    regex = r"^([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d),\s([0-1]?[0-9]|2?[0-3]):([0-5]\d) - ([a-zA-Z ()-+0-9]*):((.|\n)+?(?=((^([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d),\s([0-1]?[0-9]|2?[0-3]):([0-5]\d))|\Z)))"
    intervention_datetimes = re.findall(regex, history, re.MULTILINE)
    format = 1
    print("First match: " + str(len(intervention_datetimes)))
    if len(intervention_datetimes) == 0:
        regex = r"^\[([0-1]?[0-9]|2?[0-3]):([0-5]\d),\s([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d)\]\s([a-zA-Z ()-+0-9]*):((.|\n)+?(?=(\[|\Z)))"
        intervention_datetimes = re.findall(regex, history, re.MULTILINE)
        format = 2
        print("Second match: " + str(len(intervention_datetimes)))
    dates = []
    active_participants = []
    analysis_words = we_dict.split("\n")
    df = pd.DataFrame(columns=["total"])
    analysis_words.append("total")
    print(analysis_words)
    print(str(intervention_datetimes))
    words_df = pd.DataFrame(columns=analysis_words)
    for item in intervention_datetimes:
        text = item[8]
        participant_name = item[7]
        if participant_name not in active_participants:
            active_participants.append(participant_name)
            print(participant_name)
            df[participant_name] = [] if len(dates) == 0 else [0] * len(dates)
        print("Item " + str(item))
        if format == 1:
            day = (
                item[0]
                if len(item[0]) > 0
                else (item[1] if len(item[1]) > 0 else item[2])
            )
            full_date = day + "/" + item[3] + "/" + item[4]
        else:
            day = (
                item[2]
                if len(item[2]) > 0
                else (item[3] if len(item[3]) > 0 else item[4])
            )
            full_date = day + "/" + item[5] + "/" + item[6]
        if full_date not in dates:
            df.loc[full_date] = [0 for i in range(len(active_participants) + 1)]
            if len(analysis_words) > 0:
                words_df.loc[full_date] = [0 for i in range(len(analysis_words))]
            dates.append(full_date)
        df[participant_name].loc[[full_date]] += 1
        df["total"].loc[full_date] += 1
        for word in analysis_words:
            if word in text:
                words_df[word].loc[full_date] += 1
                words_df["total"].loc[full_date] += 1

    # sort the array
    dates = sorted(dates, key=lambda x: x[1])

    # formatted_dates = []
    # for key, value in dates.items():
    #     print(key, value)
    #     formatted_dates.append([key, value])
    # print(formatted_dates)
    num_of_questions = history.count("?")
    bar_plot = px.bar(df, x=df.index, y="total")
    stacked_bar_plot = px.bar(df, x=df.index, y=active_participants, barmode="stack")
    word_bar_plot = px.bar(df, x=words_df.index, y="total")
    proportion = "{:2.{decimal}f}%".format(
        (len(active_participants) / int(num_people) if num_people != "" else 0) * 100,
        decimal=1,
    )
    # df["Date"] = df.index
    df.insert(0, "Date", df.index)
    words_df.insert(0, "Date", words_df.index)
    # words_df["Date"] = words_df.index
    return (
        df,
        bar_plot,
        stacked_bar_plot,
        proportion,
        words_df,
        word_bar_plot,
        str(num_of_questions),
    )


with gr.Blocks() as demo:
    gr.Label("Chat history analysis")
    chat_history = gr.Textbox(
        "",
        interactive=True,
        max_lines=500,
        lines=10,
        label="Chat history",
    )
    num_of_people = gr.Textbox(
        "",
        interactive=True,
        max_lines=1,
        lines=1,
        label="Total number of people in the chat",
    )
    we_dictionary = gr.Textbox(
        "",
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
    output_interventions_plot_stacked = gr.Plot()
    output_proportion = gr.Textbox(
        "Proportion of 'we' words will be displayed here",
        label="Proportion of active usage",
    )
    output_we_words = gr.DataFrame(headers=["Date", "Number of words usage"])
    output_we_words_plot = gr.Plot()
    output_num_of_question_marks = gr.Textbox(
        "Question marks usage will be displayed here",
        label="Number of question marks",
    )
    submit_btn.click(
        process,
        inputs=[chat_history, num_of_people, we_dictionary],
        outputs=[
            output_interventions,
            output_interventions_plot,
            output_interventions_plot_stacked,
            output_proportion,
            output_we_words,
            output_we_words_plot,
            output_num_of_question_marks,
        ],
    )

    gr.Label("Output")


demo.launch(server_name="0.0.0.0", server_port=8080)

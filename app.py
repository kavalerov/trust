import re

import gradio as gr
import numpy as np
import pandas as pd
from plotly import express as px


def process(history, num_people, we_dict):
    print(history)
    regex = r"^([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d),\s([0-1]?[0-9]|2?[0-3]):([0-5]\d) - ([a-zA-Z ]*):"
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
    stacked_bar_plot = px.bar(df, x=df.index, y=active_participants, barmode="stack")

    proportion = "{:2.{decimal}f}%".format(
        (len(active_participants) / int(num_people) if num_people != "" else 0) * 100,
        decimal=1,
    )

    return (
        df,
        bar_plot,
        stacked_bar_plot,
        proportion,
        "",
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
            output_interventions_plot_stacked,
            output_proportion,
            output_we_words,
            output_num_of_question_marks,
        ],
    )

    gr.Label("Output")


demo.launch()

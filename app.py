import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from plotly import express as px
from supabase import Client, create_client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


def get_dictionaries_titles() -> List[str]:
    # get the dictionaries
    dictionaries = supabase.from_("dictionaries").select("title").execute()
    print("Getting new titles")
    return [d["title"] for d in dictionaries.data]


def show_dictionary(title: str) -> str:
    # get the dictionary
    dictionary = (
        supabase.from_("dictionaries").select("words").eq("title", title).execute()
    )
    return dictionary.data[0]["words"]


def save_dictionary(title: str, words: str) -> str:
    # get the dictionary
    dictionary = (
        supabase.from_("dictionaries").select("words").eq("title", title).execute()
    )
    if len(dictionary.data) == 0:
        dictionary = (
            supabase.from_("dictionaries")
            .insert({"title": title, "words": words})
            .execute()
        )
    else:
        dictionary = (
            supabase.from_("dictionaries")
            .update({"words": words})
            .eq("title", title)
            .execute()
        )
    return dictionary.data[0]["words"]


@dataclass
class Message:
    """
    Message contains the following:
    {
        "day": "08",
        "month": "07",
        "year": "2020",
        "time": "08:54",
        "person": "Person A",
        "content": "Hi Sarah (from Street to Scale)! We've added you to our Jarsquad WhatsApp for our new Street-To-Scale project..."
    }
    """

    date: date
    person: str
    content: str


def parse_messages(history: str) -> List[Message]:
    regex = [
        r"^([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d),\s([0-1]?[0-9]|2?[0-3]):([0-5]\d) - ([a-zA-Z ()-+0-9]*):((.|\n)+?(?=((^([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d),\s([0-1]?[0-9]|2?[0-3]):([0-5]\d))|\Z)))",  # noqa: E501
        r"^\[([0-1]?[0-9]|2?[0-3]):([0-5]\d),\s([1-9]|([012][0-9])|(3[01]))\/([0]{0,1}[1-9]|1[012])\/(\d\d\d\d)\]\s([a-zA-Z ()-+0-9]*):((.|\n)+?(?=(\[|\Z)))",
        r"\[(\d{2}\/\d{2}\/\d{4}), (\d{2}:\d{2}:\d{2})\] ([^:]+) ?: (.*?)(?=\n\[\d{2}\/\d{2}\/\d{4}, \d{2}:\d{2}:\d{2}\]|$)",
    ]

    format = None
    messages: List[str] = []
    for index, r in enumerate(regex):
        print("Trying regex " + str(index))
        messages = re.findall(r, history, re.MULTILINE)
        print("Matched " + str(len(messages)))
        if len(messages) > 0:
            format = index
            break
    print(format)
    if format is None:
        raise Exception("No regex matched")

    if format == 0:
        return [
            Message(
                date=date(
                    int(message[4]),
                    int(message[3]),
                    int(
                        message[0]
                        if len(message[0]) > 0
                        else (message[1] if len(message[1]) > 0 else message[2])
                    ),
                ),
                person=message[7],
                content=message[8],
            )
            for message in messages
        ]
    elif format == 1:
        return [
            Message(
                date=date(
                    int(message[6]),
                    int(message[5]),
                    int(
                        message[2]
                        if len(message[2]) > 0
                        else (message[3] if len(message[3]) > 0 else message[4])
                    ),
                ),
                person=message[7],
                content=message[8],
            )
            for message in messages
        ]
    elif format == 2:
        return [
            Message(
                date=datetime.strptime(message[0], "%d/%m/%Y").date(),
                person=message[2],
                content=message[3],
            )
            for message in messages
        ]
    else:
        raise Exception("No regex matched")


def process(history, num_people, we_dict):
    messages = parse_messages(history)
    print(messages)
    messages = sorted(messages, key=lambda x: x.date)
    print(messages)
    start_date = messages[0].date
    end_date = messages[-1].date
    active_participants = list(set([message.person for message in messages]))
    num_of_days = (end_date - start_date).days + 1
    date_index = pd.date_range(start_date, end_date, periods=num_of_days)
    date_index = date_index.date
    print("Start date: " + str(start_date))
    print("End date: " + str(end_date))
    print(date_index)

    # Calculate number of messages by person per day and total
    num_of_messages_df = pd.DataFrame(
        index=date_index, columns=active_participants, dtype=int
    )
    num_of_messages_df = num_of_messages_df.fillna(0)
    for message in messages:
        num_of_messages_df.loc[message.date][message.person] += 1
    num_of_messages_df["total"] = num_of_messages_df.sum(axis=1)
    print(num_of_messages_df)

    # Generate bar plot of number of messages by person per day and total
    num_of_messages_bar_plot = px.bar(
        num_of_messages_df, x=num_of_messages_df.index, y="total"
    )

    # Generate stacked bar plot of number of messages by person per day
    num_of_messages_stacked_bar_plot = px.bar(
        num_of_messages_df,
        x=num_of_messages_df.index,
        y=active_participants,
        barmode="stack",
    )

    # Calculate number of question marks in the history
    num_of_questions = history.count("?")

    # Calculate proportion of active participants
    proportion = "{:2.{decimal}f}%".format(
        (len(active_participants) / int(num_people) if num_people != "" else 0) * 100,
        decimal=1,
    )

    # Calculate number of words from the dictionary per day and total
    analysis_words = we_dict.split("\n")
    words_df = pd.DataFrame(index=date_index, columns=analysis_words, dtype=int)
    words_df = words_df.fillna(0)
    for message in messages:
        for word in analysis_words:
            count = message.content.count(word)
            words_df.loc[message.date][word] += count

    words_totals = words_df.sum(axis=0)

    # Generate stacked bar plot of number of words from the dictionary per day and total
    word_stacked_bar_plot = px.bar(
        words_df, x=words_df.index, y=analysis_words, barmode="stack"
    )

    # df.insert(0, "Date", df.index)
    # words_df.insert(0, "Date", words_df.index)

    return (
        num_of_messages_df,
        num_of_messages_bar_plot,
        num_of_messages_stacked_bar_plot,
        proportion,
        words_df,
        words_totals,
        word_stacked_bar_plot,
        str(num_of_questions),
    )


with gr.Blocks() as demo:
    gr.Label("Chat history analysis")
    dict_list = get_dictionaries_titles()
    print(dict_list)
    with gr.Tab("Analysis") as analysis_tab:
        dict_reload = gr.Button("Reload dictionaries")
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
        dict_dropdown = gr.Dropdown(
            dict_list, label="Dictionary of words to check for occurance"
        )
        we_dictionary = gr.Textbox(
            "",
            interactive=False,
            max_lines=500,
            lines=10,
            label="Full dictionary. Please go to the 'Dictionaries' tab to edit a dictionary, or 'Add Dictionaries' to add a new one.",
        )
        dict_reload.click(get_dictionaries_titles, inputs=[], outputs=[dict_dropdown])

        dict_dropdown.change(
            show_dictionary,
            inputs=[dict_dropdown],
            outputs=[we_dictionary],
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
        output_we_words_totals = gr.DataFrame(headers=["Word", "Total usage"])
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
                output_we_words_totals,
                output_we_words_plot,
                output_num_of_question_marks,
            ],
        )
    with gr.Tab("Dictionaries") as dictionaries_tab:
        gr.Label(
            "Choose dictionary from the list to see the words. Go to 'Edit Dictionaries' to add new ones."
        )
        # get list of dictionaries from supabase
        dict_reload = gr.Button("Reload dictionaries")
        dictionary_list = gr.Dropdown(dict_list, label="Dictionaries")
        dict_reload.click(get_dictionaries_titles, inputs=[], outputs=[dictionary_list])
        dictionary_content = gr.Textbox(
            "",
            interactive=True,
            max_lines=500,
            lines=10,
            label="Dictionary content",
        )
        btn_save_dictionary = gr.Button(value="Save")
        btn_save_dictionary.click(
            save_dictionary,
            inputs=[dictionary_list, dictionary_content],
            outputs=[dictionary_content],
        )

        dictionary_list.change(
            show_dictionary,
            inputs=[dictionary_list],
            outputs=[dictionary_content],
        )
    with gr.Tab("Add Dictionaries"):
        gr.Label(
            "Choose dictionary from the list to see the words. Go to 'Edit Dictionaries' to add new ones."
        )

        dictionary_name = gr.Textbox(
            "",
            interactive=True,
            max_lines=1,
            lines=1,
            label="New dictionary name",
        )
        dictionary_content = gr.Textbox(
            "",
            interactive=True,
            max_lines=500,
            lines=10,
            label="Dictionary content",
        )
        btn_save_dictionary = gr.Button(value="Save")
        btn_save_dictionary.click(
            save_dictionary,
            inputs=[dictionary_name, dictionary_content],
            outputs=[dictionary_content],
        )

    with gr.Tab("About"):
        gr.Label("Chat history analysis")
        gr.Label("Todo: how to use this tool")


demo.launch(server_name="0.0.0.0", server_port=8080)

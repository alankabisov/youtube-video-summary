import os


import streamlit as st
from urllib.parse import urlparse, parse_qs

from tqdm import tqdm
from stqdm import stqdm

# https://github.com/pytorch/pytorch/issues/77764
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from youtube_transcript_api import YouTubeTranscriptApi

from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

import torch

# Setting device for PYTorch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')



def get_videoid_from_url(url:str):
    url_data = urlparse(url)
    query = parse_qs(url_data.query)

    try:
        video_id = query["v"][0]
    except KeyError:
        video_id = ''

    return video_id

def process_click_callback():
    st.session_state.process_btn = True

    print('Using {} device'.format(device))

    transcript_list = YouTubeTranscriptApi.list_transcripts('aircAruvnKk')  # 3blue1Brown

    try:
        transcript = transcript_list.find_manually_created_transcript(['en'])
    except Exception as e:
        print('No manual transcripts were found, trying to load generated ones...')
        transcript = transcript_list.find_generated_transcript(['en'])

    subtitles = transcript.fetch()

    subtitles = [sbt['text'] for sbt in subtitles if sbt['text'] != '[Music]']
    subtitles_len = [len(sbt) for sbt in subtitles]
    sbt_mean_len = sum(subtitles_len)/len(subtitles_len)

    print('Mean length of subtitles: {}'.format(sbt_mean_len))
    print(subtitles)
    print(len(subtitles))

    # Number of subtitles per step/summary
    # Since number length of transcripts differs
    # between generated and manual ones
    # we set different step size
    n_sbt_per_step = int(400 / (sbt_mean_len / 4))
    print('Number subtitles per summary: {}'.format(n_sbt_per_step))

    n_steps = len(subtitles) // n_sbt_per_step if len(subtitles) % n_sbt_per_step == 0 else \
        len(subtitles) // n_sbt_per_step + 1

    summaries = []

    for i in stqdm(range(n_steps)):
        sbt_txt = ' '.join(subtitles[n_sbt_per_step*i:n_sbt_per_step*(i+1)])
        # print('length of text: {}'.format(len(sbt_txt)))
        # print(sbt_txt)

        summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small',
                              max_length=512, truncation=True)

        summary = summarizer(sbt_txt, do_sample=False)
        summary = summary[0]['summary_text']

        # print('Summary: ' + summary)
        summaries.append(summary)

    out = ' '.join(summaries)
    print(out)

    st.session_state.summary_output = out
    st.success('Processing complete!', icon="✅")
    st.session_state.process_btn = False



def main():
    st.title('YouTube Video Summary 📃')
    st.markdown('Creates summary for given YouTube video URL based on transcripts.')
    st.code('https://www.youtube.com/watch?v=aircAruvnKk')
    st.code('https://youtu.be/p0G68ORc8uQ')

    col1, col2 = st.columns(2)

    with col1:
        video_url = st.text_input('YouTube Video URL:',  placeholder='YouTube URL',
                                 label_visibility='collapsed')
        st.write(get_videoid_from_url(video_url))

    with col2:
        st.button('Process 📭', key='process_btn', on_click=process_click_callback)

    st.text_area(label='', key='summary_output', height=444)






    # x = st.slider('Select a value')
    # st.write(x, 'squared is', x * x)


if __name__ == "__main__":
    main()
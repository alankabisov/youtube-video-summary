import os

import streamlit as st

from urllib.parse import urlparse, parse_qs

from stqdm import stqdm

# https://github.com/pytorch/pytorch/issues/77764
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from transformers import pipeline

import torch

# Setting device for PyTorch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.has_mps:
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class InvalidURLException(Exception):
    pass


def get_videoid_from_url(url: str):
    '''
    Gets video ID from give YouTube video URL

    :param url: YouTube video URL in 2 formats (standard and short)
    :return: id of YouTube video
    :raises InvalidURLException: If URL is not valid
    '''
    url_data = urlparse(url)
    query = parse_qs(url_data.query)

    if ('v' in query) & ('youtube.com' in url_data.netloc):
        video_id = query["v"][0]
    elif 'youtu.be' in url_data.netloc:
        path_lst = url.split('/')

        if path_lst:
            video_id = path_lst[-1]
        else:
            raise InvalidURLException('Invalid URL')
    else:
        raise InvalidURLException('Invalid URL')

    return video_id


def get_transcripts(url: str):
    '''
    Loads transcripts for given URL

    :param url: YouTube video URL
    :return: list, list of subtitles
    '''

    video_id = get_videoid_from_url(video_url_inp)

    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    try:
        transcript = transcript_list.find_manually_created_transcript(['en'])
    except NoTranscriptFound as e:
        # Doesn't work on HuggingFace v1.1.0 version
        # st.info('No manual transcripts were found, trying to load generated ones...')
        transcript = transcript_list.find_generated_transcript(['en'])

    subtitles = transcript.fetch()

    subtitles = [sbt['text'] for sbt in subtitles if sbt['text'] != '[Music]']

    return subtitles


def generate_summary(subtitles: list):
    '''
    Creates summary based on subtitles of YouTube video.

    Uses T5-small model which shows best results for different topics
    of videos.

    :param subtitles: list of subtitles strings
    :return: summary based on subtitles
    '''
    subtitles_len = [len(sbt) for sbt in subtitles]
    sbt_mean_len = sum(subtitles_len) / len(subtitles_len)

    # Number of subtitles per step/summary
    # Since number length of transcripts differs
    # between generated and manual ones
    # we set different step size
    n_sbt_per_step = int(400 / (sbt_mean_len / 4))

    n_steps = len(subtitles) // n_sbt_per_step if len(subtitles) % n_sbt_per_step == 0 else \
        len(subtitles) // n_sbt_per_step + 1

    summaries = []

    for i in stqdm(range(n_steps)):
        sbt_txt = ' '.join(subtitles[n_sbt_per_step * i:n_sbt_per_step * (i + 1)])

        summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small',
                              max_length=512, truncation=True)

        summary = summarizer(sbt_txt, do_sample=False)
        summary = summary[0]['summary_text']

        summaries.append(summary)

    return ' '.join(summaries)


def process_click_callback():
    '''
    Callback for process button click
    '''
    global is_processing

    if is_processing:
        return
    else:
        is_processing = True

    global video_url_inp

    try:
        subtitles = get_transcripts(video_url_inp)
    except InvalidURLException as iue:
        is_processing = False
        st.error('Invalid YouTube URL, please provide URL in format that is shown on Examples')
        st.experimental_rerun()
    except TranscriptsDisabled as tde:
        is_processing = False
        st.error('Could not retrieve a transcript for given ID')
        st.experimental_rerun()

    summary = generate_summary(subtitles)

    st.session_state.summary_output = summary
    st.success('Processing complete! ✅')

    is_processing = False


if __name__ == "__main__":
    # State of processing
    is_processing = False

    st.title('YouTube Video Summary 📃')
    st.markdown('Creates summary for given YouTube video URL based on transcripts.')
    st.code('https://www.youtube.com/watch?v=skl4OXNA12U')
    st.code('https://youtu.be/mEQc-iAbEBk')

    col1, col2 = st.columns(2)

    with col1:
        video_url_inp = st.text_input('', placeholder='YouTube URL')

    with col2:
        st.markdown('#') # Adds empty space
        process_btn = st.button('🗜️Process', key='process_btn', on_click=process_click_callback)

    summary_out_txt = st.text_area(label='', key='summary_output', height=400)

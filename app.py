import streamlit as st
from urllib.parse import urlparse, parse_qs



def get_videoid_from_url(url:str):
    url_data = urlparse(url)
    query = parse_qs(url_data.query)

    try:
        video_id = query["v"][0]
    except KeyError:
        video_id = ''

    return video_id


def main():
    st.title('YouTube Video Summary 📃')
    st.text('This app creates summary for given YouTube video based on transcripts.')

    col1, col2 = st.columns(2)

    with col1:
        video_id = st.text_input('YouTube Video ID:', placeholder='Live it empty if you want to use example video...')
        # video_id = 'aircAruvnKk'
        # video_id = 'https://www.youtube.com/watch?v=aircAruvnKk'
        st.write(get_videoid_from_url(video_id))

    with col2:
        st.button('Process')



    # x = st.slider('Select a value')
    # st.write(x, 'squared is', x * x)


if __name__ == "__main__":
    main()
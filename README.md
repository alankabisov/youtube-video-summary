---
title: Youtube Video Summary
emoji: 📝
colorFrom: purple
colorTo: red
sdk: streamlit
sdk_version: 1.10.0
app_file: app.py
models: t5-small
pinned: false
---

# YouTube Video Summary 📝
Extracts transcripts for given video URL and creates summary. Uses [T5-small](https://huggingface.co/t5-small) model 
under the hood since it have shown the best results on general purpose tasks.   

**Online demo:** [🤗 Spaces](https://huggingface.co/spaces/alankabisov/youtube-video-summary)

### Requirements
```
torch
transformers
youtube_transcript_api
tqdm
stqdm
streamlit==1.10.0
```


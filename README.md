# Visual-Question-Answering
A Visual Question Answering Model with a Simple GUI that deals with both Images and Videos

## Dataset
The dataset used is the publicly available [VQA Dataset](https://visualqa.org/).  
VQA is a new dataset containing open-ended questions about images. These questions require an understanding of vision, language and commonsense knowledge to answer.
- 265,016 images (COCO and abstract scenes)
- At least 3 questions (5.4 questions on average) per image
- 10 ground truth answers per question
- 3 plausible (but likely incorrect) answers per question
- Automatic evaluation metric.  

We used the Version 2 of the dataset.

## Architecture
### Encoder
We used OpenAI's CLIP Encoder to encode the images and the questions in an equal embeddings space.

### Decoder
GPT-2 is used as a decoder to convert the encoded embeddings back to sequence whilst comparing it with the 'Ground Truth' answers of the images.

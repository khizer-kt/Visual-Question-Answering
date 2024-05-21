from model import VQAModel
import os 
import torch
import clip
from PIL import Image

class Inferece():
    def __init__(self,pathCheckpoint) -> None:

        assert os.path.isfile(pathCheckpoint), f"No checkpoint found at {pathCheckpoint}"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VQAModel()
        chkpoint = torch.load(pathCheckpoint)
        self.model.load_state_dict(chkpoint["model_state_dict"])
        self.clip_encoder, self.preprocess = clip.load("ViT-B/32",device=self.device)


    def query(self,pathToImg,textQuery):
        assert os.path.isfile(pathToImg) and type(textQuery) is str , "Expected Image path and query in form of text"

        pilImg = Image.open(pathToImg).convert('L')

        img = self.preprocess(pilImg).unsqueeze(0).to(self.device)

        question = clip.tokenize(question["question"]).to(self.device)


        with torch.no_grad():
            img_features = self.clip_encoder.encode_image(img).squeeze()
            question_features = self.clip_encoder.encode_text(question).squeeze()
        
        fused_input = torch.cat((img_features,question_features),dim=1).to(torch.float32)
        fused_input  = fused_input.unsqueeze(0) # adding batch dim

        outputs, _ = self.model(fused_input)

        print("Output = ",outputs)

        return outputs




        
        


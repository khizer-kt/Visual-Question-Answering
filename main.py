from inference import Inferece

if __name__ == "__main__":

    infer = Inferece(pathCheckpoint="./checkpoint.pth")

    infer.query("./cry.jpeg","what is child doing ?")
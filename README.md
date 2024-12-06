"# WhoLooksLikeMe" 

### Basic Setup and Testing
- Python 3.11 works with this setup - Tested
- Change environemnt to that : Goto System ? Environmental Variables and update path to point to python311
- run python ./setupfacenet/setup-vggface2.py from the WhoLooksLikeMe folder
- activate the new tensorflow-vggface venv By going to venv/tensorflow_facenet/Scripts and then run ./activate
#### Download the inception v3 weights for setting up the pretrained model. 
- Go to https://www.kaggle.com/models/google/inception-v3
- Download and paste the folder in inception-v3 folder. The contents inside of the folder should be "saved_model.pb" and a variables folder.
- This is being used by **Facenet/FaceNetModel.py** to load the pretrained model.
- Open Facenet folder / Tester.ipynb and run the code after replacing the image path you want to test. 



### To train and create more embeddings. 
#### To download the dataset from hugging face and setup the images in your local repo 
- Open DataCleaners/Preprocess.ipynb and run it.
- Open Preprocess.ipynb and run the ipynb file to generate the dataset_images folder which can be used for training
- After creating the dataset_images open Training.ipynb file.
- Make sure that label_names.csv and output_with_gender.csv file are present inside DataCleaners folder


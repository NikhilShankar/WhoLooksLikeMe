"# WhoLooksLikeMe" 

#### Basic Setup and Testing
- Python 3.11 works with this setup - Tested
- Change environemnt to that : Goto System ? Environmental Variables and update path to point to python311
- run python ./setupfacenet/setup-vggface2.py from the WhoLooksLikeMe folder
- activate the new tensorflow-vggface venv By going to venv/tensorflow_facenet/Scripts and then run ./activate
- Open Facenet folder / Tester.ipynb and run the code after replacing the image path you want to test. 


#### To train and create more embeddings. 
- Open Preprocess.ipynb and run the ipynb file to generate the dataset_images folder which can be used for training
- After creating the dataset_images open Training.ipynb file.
- Make sure that label_names.csv and output_with_gender.csv file are present.


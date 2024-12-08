to run
uvicorn app.main:app --reload


to install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn python-docx spacy scikit-learn transformers
python -m spacy download en_core_web_sm


You’ll create two utility files: document_parser.py for extracting text from .docx and text_processor.py for preprocessing and comparing text.

You’ll create a new FastAPI router (compare_resume.py) to handle the API request to compare the resume and job description.

app/routers/compare_resume.py

packages
annotated-types   0.7.0
anyio             4.6.2.post1
asttokens         2.4.1
certifi           2024.8.30
colorama          0.4.6
comm              0.2.2
debugpy           1.8.5
decorator         5.1.1
distro            1.9.0
executing         2.1.0
h11               0.14.0
httpcore          1.0.7
httpx             0.27.2
idna              3.10
ipykernel         6.29.5
ipython           8.27.0
jedi              0.19.1
jiter             0.7.1
jupyter_client    8.6.2
jupyter_core      5.7.2
matplotlib-inline 0.1.7
nest-asyncio      1.6.0
openai            1.55.0
packaging         24.1
parso             0.8.4
pip               24.0
platformdirs      4.3.2
prompt_toolkit    3.0.47
psutil            6.0.0
pure_eval         0.2.3
pydantic          2.10.1
pydantic_core     2.27.1
Pygments          2.18.0
python-dateutil   2.9.0.post0
pywin32           306
pyzmq             26.2.0
setuptools        65.5.0
six               1.16.0
sniffio           1.3.1
stack-data        0.6.3
tornado           6.4.1
tqdm              4.67.0
traitlets         5.14.3
typing_extensions 4.12.2
wcwidth           0.2.13
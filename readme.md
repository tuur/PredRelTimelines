## Code associated with:

### Temporal Information Extraction by Predicting Relative Time-lines
Artuur Leeuwenberg & Marie-Francine Moens
*In Proceedings of EMNLP*, Brussels, Belgium, 2018.

#### Terms of Usage
Apache License 2.0
> Please cite the paper when using the code.

### What this code can be used for:
Training relative time-line prediction models (TLM), trained from TimeML-annotated data, and to convert TimeML-style annotations to relative time-lines (TL2RTL), i.e. to reproduce the results from the EMNLP 2018 paper. The data used in the original paper was added to facilitate reproducing the results from the paper.

### How do I get set up? ###
1. Install  Python 3.5 (if not yet installed)
2. Setup and activate (line 2) a virtual environment, and install the Python dependencies with:
```
virtualenv venv -p python3.5
source venv/bin/activate
pip install -r requirements.txt
```
3. Unzip the word vectors in `data/wordvectos/xaa` to obtain `data/wordvectors/glove.6B.50d.w2vf.txt`

### Reproducing the results from the paper:
To produce the results from Table 4 from the paper, you need to run two scripts:
```
sh EXP-TLM.sh
sh EXP-TL2RTL.sh
```
The first will rerun the S-TLM and C-TLM experiments in the different loss settings (evaluation scores can be found at the end of the corresponding log files). And the second will rerun the TL2RTL experiments.

`!!!` By default these scripts run many subprocesses simultaneously. If you prefer to run the experiments one-by-one you should remove the &-symbol at the end of the python calls in the scripts. You can indicate which GPU you want to use by setting the CUDA variable in the scripts (-1 for using CPU instead).

### Obtaining more information 

To get more information on how to use the scripts you can inspect the .sh files or run:
```
python main.py -h
```
for TLM options or 
```
python timeml_to_timeline.py -h
```
for TL2RTL options.

### Questions?
> Any questions? Feel free to send me an email!

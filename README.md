# Discord-Bot-MaliciousURL-Detection
Deep Learning TCN (Temporal Convolutional Networks) based phishing detection bot for Discord  
Codebase originally from https://github.com/Dinnerspy/Discord-Bot-Phishing-Detection. and heavily modified to use TCN instead

## Requirements
Built with Python 3.8.0
```
beautifulsoup4==4.11.1
discord.py==1.7.3
dnspython==2.2.1
emoji==2.0.0
numpy==1.22.3
pandas==1.4.2
python_Levenshtein==0.12.2
requests==2.23.0
scikit_learn==1.1.1
tldextract==3.3.1
whois==0.9.16
```
Discord API token can be retrevied from [here](https://discord.com/developers/applications).  
Openpagerank API token can be retrevied from [here](https://www.domcop.com/openpagerank/auth/signup).
## Install
```
pip install -r /path/to/requirements.txt
Place data folder in same directory as the python scripts
Insert token for Discord Bot API in FishFinder.py
Insert token for Openpagerank API in FeatureExtractor.py
Python FishFinder.py
```
## Usage 
to be added






## Acknowledgement
Hannousse, Abdelhakim; Yahiouche, Salima (2021), “Web page phishing detection”, Mendeley Data, V3, doi: 10.17632/c2gw7fy2j4.3 also available here:(https://data.mendeley.com/datasets/c2gw7fy2j4/3)
N. Ouellette, Y. Baseri, and B. Kaur, “I Am Bot the ‘Fish Finder’: Detecting Malware that Targets Online Gaming Platform,” in Proceedings of Eighth International Congress on Information and Communication Technology, vol. 693, X.-S. Yang, R. S. Sherratt, N. Dey, and A. Joshi, Eds., in Lecture Notes in Networks and Systems, vol. 693. , Singapore: Springer Nature Singapore, 2023, pp. 261–274. doi: 10.1007/978-981-99-3243-6_21.

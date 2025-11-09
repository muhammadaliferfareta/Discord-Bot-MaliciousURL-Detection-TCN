import FeatureExtractor as FeatureE
import discord
from discord import Intents
import emoji
import asyncio
from functools import partial
import tldextract
import numpy as np
import os
import pandas as pd
import re as regex
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
ChannelName ="PhishfinderLogs"
#clock yellow orange green red
emojis = [emoji.emojize(':stopwatch:'), emoji.emojize(':yellow_square:'), emoji.emojize(':orange_square:'), emoji.emojize(':green_square:'),emoji.emojize(':red_square:'),emoji.emojize(":cross_mark:")]
features1 = [
    "length_url",
    "nb_eq",
    "nb_underscore",
    "nb_www",
    "http_in_path",
    "ratio_digits_url",
    "ratio_digits_host",
    "port",
    "shortening_service",
    "char_repeat",
    "longest_word_path",
    "phish_hints",
    "domain_in_brand",
    "suspecious_tld",
    "nb_hyperlinks",
    "ratio_intHyperlinks",
    "ratio_extRedirection",
    "safe_anchor",
    "right_clic",
    "empty_title",
    "domain_in_title",
    "domain_with_copyright",
    "domain_registration_length",
    "domain_age",
    "web_traffic",
    "dns_record",
    "google_index",
    "page_rank",
    "status",
]

features2 = [
    "length_hostname",
    "nb_dots",
    "nb_hyphens",
    "nb_qm",
    "nb_underscore",
    "nb_space",
    "nb_www",
    "ratio_digits_host",
    "length_words_raw",
    "longest_word_path",
    "avg_words_raw",
    "phish_hints",
    "nb_hyperlinks",
    "google_index",
    "page_rank",
    "status",
]

features3 = [
'length_url', 
'length_hostname', 
'ip', 
'nb_qm', 
'nb_eq', 
'nb_slash', 
'nb_www', 
'ratio_digits_url', 
'phish_hints', 
'nb_hyperlinks', 
'ratio_intHyperlinks', 
'domain_in_title', 
'domain_age', 
'google_index', 
'page_rank',
'status',
]

features4 = [
    "length_hostname",
    "ip",
    "nb_hyphens",
    "nb_qm",
    "nb_underscore",
    "nb_space",
    "nb_www",
    "shortening_service",
    "avg_words_raw",
    "avg_word_path",
    "phish_hints",
    "nb_hyperlinks",
    "ratio_extHyperlinks",
    "ratio_extRedirection",
    "ratio_intMedia",
    "ratio_extMedia",
    "safe_anchor",
    "domain_age",
    "google_index",
    "page_rank",
    "status",
]
featurestoget = [
'length_url',
'length_hostname',
'ip',
'nb_dots',
'nb_hyphens',
'nb_qm',
'nb_eq',
'nb_underscore',
'nb_slash',
'nb_space',
'nb_www',
'http_in_path',
'ratio_digits_url',
'ratio_digits_host',
'port',
'shortening_service',
'char_repeat',
'longest_word_path',
'length_words_raw',
'avg_words_raw',
'avg_word_path',
'phish_hints',
'domain_in_brand',
'suspecious_tld',
'nb_hyperlinks',
'ratio_extHyperlinks',
'ratio_intHyperlinks',
'ratio_extRedirection',
'ratio_intMedia',
'ratio_extMedia',
'safe_anchor',
'right_clic',
'empty_title',
'domain_in_title',
'domain_with_copyright',
'domain_registration_length',
'domain_age',
'web_traffic',
'dns_record',
'google_index',
'page_rank',
]




pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)
dataBaseline = pd.read_csv("data/dataset_phishing.csv")
#features = dataBaseline.drop("status", 1).columns.values.tolist()
###d = {'benign': 1, 'malicious': 0}
###data['status'] = data['status'].map(d)
#normaldata = dataBaseline[features].copy()
#dataBaseline = dataBaseline.drop(features, 1)
#x = normaldata.values #returns a numpy array 
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
#x_scaled = min_max_scaler.fit_transform(x)
#normaldata = pd.DataFrame(x_scaled, columns=normaldata.columns)
#dataBaseline = pd.concat([normaldata, dataBaseline], axis=1)


dataset1 = dataBaseline[features1].copy()
dataset2 = dataBaseline[features2].copy()
dataset3 = dataBaseline[features3].copy()
dataset4 = dataBaseline[features4].copy()

temp = dataset1
X = temp.drop("status", 1)
featuretodrop1 = X.columns.values.tolist()
y = dataset1.drop(featuretodrop1, 1)
Xdataset1_train, Xdataset1_test, ydataset1_train, ydataset1_test = train_test_split(X, y, test_size=.30, random_state=42)  # Use 30% test, 70% train for better training

temp = dataset2
X = temp.drop("status", 1)
featuretodrop2 = X.columns.values.tolist()
y = dataset2.drop(featuretodrop2, 1)
Xdataset2_train, Xdataset2_test, ydataset2_train, ydataset2_test = train_test_split(X, y, test_size=.30, random_state=42)

temp = dataset3
X = temp.drop("status", 1)
featuretodrop3 = X.columns.values.tolist()
y = dataset3.drop(featuretodrop3, 1)
Xdataset3_train, Xdataset3_test, ydataset3_train, ydataset3_test = train_test_split(X, y, test_size=.30, random_state=42)

temp = dataset4
X = temp.drop("status", 1)
featuretodrop4 = X.columns.values.tolist()
y = dataset4.drop(featuretodrop4, 1)
Xdataset4_train, Xdataset4_test, ydataset4_train, ydataset4_test = train_test_split(X, y, test_size=.30, random_state=42)

def runRandomForest(Xtrain, ytrain):
    print("Training Random Forest model...")
        
    pipe = Pipeline([("scale", StandardScaler()),
                    ("rf", RandomForestClassifier())
                    ])

    param_grid = {}
    print("Starting cross-validation...")
    grid = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)  # Original: 10-fold cross-validation for better accuracy
    grid.fit(Xtrain, ytrain.values.ravel())
    print("Cross-validation complete!")

    model_best = grid.best_estimator_
    model_best.fit(Xtrain, ytrain.values.ravel())

    return model_best

def runExtraTrees(Xtrain, ytrain):
    print("Training Extra Trees model...")
    
    pipe = Pipeline([("scale", StandardScaler()),
                    ("etc", ExtraTreesClassifier())
                    ])

    param_grid = {}
    
    print("Starting cross-validation...")
    grid = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)  # Original: 10-fold cross-validation for better accuracy
    grid.fit(Xtrain, ytrain.values.ravel())
    print("Cross-validation complete!")

    model_best = grid.best_estimator_
    model_best.fit(Xtrain, ytrain.values.ravel())
    
    return model_best
    
print("\nStarting model training...")
print("Training Algorithm 1 (Random Forest)...")
Algo1 = runRandomForest(Xdataset1_train, ydataset1_train)
print("Training Algorithm 2 (Extra Trees)...")
Algo2 = runExtraTrees(Xdataset2_train, ydataset2_train)
print("Training Algorithm 3 (Random Forest)...")
Algo3 = runRandomForest(Xdataset3_train, ydataset3_train)
print("Training Algorithm 4 (Random Forest)...")
Algo4 = runRandomForest(Xdataset4_train, ydataset4_train)
print("All models trained successfully!\n")

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
TOKEN = os.getenv('DISCORD_TOKEN')  # Get token from environment variable
intents = Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def normalizeDataExt(data):
    """Normalize the feature data to match training data scale"""
    normaldata = data[featurestoget].copy()
    
    # Define known ranges for specific features
    normaldata['domain_age'] = normaldata['domain_age'].clip(lower=0, upper=10000)
    normaldata['page_rank'] = normaldata['page_rank'].clip(lower=-1, upper=10)
    normaldata['web_traffic'] = normaldata['web_traffic'].clip(lower=0, upper=100)
    
    # Standard scaling for numerical features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(normaldata.values)
    normaldata = pd.DataFrame(x_scaled, columns=normaldata.columns)


   
    return  normaldata

def check_results(answer1, answer2, answer3, answer4, url):
    # We're not using vote counting anymore, keeping function for compatibility
    return 0

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')


@client.event
async def on_message(message):
    urls = regex.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message.content.lower())
    flag = False
    flagforNetworkIssue = False
    ##Check if the log channel is there:
    Guild = message.guild
    textchannelList = Guild.text_channels
    flagChannel = True
    for x in textchannelList:
        if(x.name == ChannelName.lower()):
            flagChannel = False
    if(flagChannel):
         await Guild.create_text_channel(ChannelName)
    if(len(urls) > 0):
        flag = True
    if (flag == True):
        await message.add_reaction(emojis[0])
        print(urls[0])
        # Run blocking feature extraction in a background thread to avoid blocking the event loop
        try:
            # asyncio.to_thread was added in Python 3.9; fall back to run_in_executor on 3.8
            if hasattr(asyncio, 'to_thread'):
                features_list = await asyncio.to_thread(FeatureE.extract_features, urls[0])
            else:
                loop = asyncio.get_running_loop()
                features_list = await loop.run_in_executor(None, partial(FeatureE.extract_features, urls[0]))
        except Exception as e:
            print("Feature extraction error:", e)
            features_list = []
        urlinquestion = pd.DataFrame(features_list, columns=featurestoget) if features_list and len(features_list) > 0 else pd.DataFrame(columns=featurestoget)
#        urlinquestion = normalizeDataExt(urlinquestion)
        if features_list is None or urlinquestion.shape == (0,41):
            flagforNetworkIssue = True
        embedVar  = ""
        Channel = discord.utils.get(Guild.channels, name=ChannelName.lower())
        await message.remove_reaction(emojis[0],client.user)
        if flagforNetworkIssue == False:    
            # Predict labels
            awns1 = Algo1.predict(urlinquestion[featuretodrop1].copy())
            awns2 = Algo2.predict(urlinquestion[featuretodrop2].copy())
            awns3 = Algo3.predict(urlinquestion[featuretodrop3].copy())
            awns4 = Algo4.predict(urlinquestion[featuretodrop4].copy())

            # Get malicious probabilities where available
            probs = []
            try:
                if hasattr(Algo1, 'predict_proba'):
                    idx = list(Algo1.classes_).index('phishing') if 'phishing' in Algo1.classes_ else None
                    probs.append(Algo1.predict_proba(urlinquestion[featuretodrop1].copy())[0][idx] if idx is not None else 0.0)
                else:
                    probs.append(0.0)
            except Exception:
                probs.append(0.0)
            try:
                if hasattr(Algo2, 'predict_proba'):
                    idx = list(Algo2.classes_).index('phishing') if 'phishing' in Algo2.classes_ else None
                    probs.append(Algo2.predict_proba(urlinquestion[featuretodrop2].copy())[0][idx] if idx is not None else 0.0)
                else:
                    probs.append(0.0)
            except Exception:
                probs.append(0.0)
            try:
                if hasattr(Algo3, 'predict_proba'):
                    idx = list(Algo3.classes_).index('phishing') if 'phishing' in Algo3.classes_ else None
                    probs.append(Algo3.predict_proba(urlinquestion[featuretodrop3].copy())[0][idx] if idx is not None else 0.0)
                else:
                    probs.append(0.0)
            except Exception:
                probs.append(0.0)
            try:
                if hasattr(Algo4, 'predict_proba'):
                    idx = list(Algo4.classes_).index('phishing') if 'phishing' in Algo4.classes_ else None
                    probs.append(Algo4.predict_proba(urlinquestion[featuretodrop4].copy())[0][idx] if idx is not None else 0.0)
                else:
                    probs.append(0.0)
            except Exception:
                probs.append(0.0)

            # Calculate average probability
            avg_phish_prob = sum(probs) / len(probs) if len(probs) > 0 else 0.0
            print("Preds:", awns1[0], awns2[0], awns3[0], awns4[0])
            print("Avg phish prob:", round(avg_phish_prob, 3))

            # Calculate malicious confidence as percentage
            malicious_confidence = round(avg_phish_prob * 100, 1)
            
            if flagforNetworkIssue:
                embedVar = discord.Embed(
                    title='Unable to analyze '+urls[0].replace("http://","").replace("https://",""),
                    description='Could not access or analyze this URL. Treat it as suspicious!',
                    color=0x660000,
                    type='rich'
                )
                embedVar.add_field(name="Error", value="Failed to extract URL features - the site may be blocking automated access or could be offline", inline=False)
                await message.add_reaction(emojis[5])
            else:
                # Set status and format based on confidence
                if malicious_confidence < 50:
                    color = 0x00FF00  # Green
                    status = "BENIGN"
                    description = f"This URL appears safe (Confidence: {100-malicious_confidence}%)"
                    await message.add_reaction(emojis[3])
                elif malicious_confidence >= 90:
                    color = 0x000000  # Black
                    status = "HIGH RISK"
                    description = f"⚠️ High risk malicious URL detected (Confidence: {malicious_confidence}%)"
                    await message.delete()  # Delete high-risk messages
                else:
                    color = 0xFF9900  # Orange
                    status = "SUSPICIOUS"
                    description = f"⚠️ Potentially malicious URL (Confidence: {malicious_confidence}%)"
                    await message.add_reaction(emojis[2])
                
                # Create embed with results
                embedVar = discord.Embed(
                    title=f"{status}: {urls[0].replace('http://','').replace('https://','')}",
                    description=description,
                    color=color,
                    type='rich'
                )

        await Channel.send(embed=embedVar)
        #await Channel.send(+ 'Algorithm 1: '+awns1[0]+ '\n'+ 'Algorithm 2: '+awns2[0]+ '\n'+ 'Algorithm 3: '+awns3[0]+ '\n'+ 'Algorithm 4: '+awns4[0])

@client.event
async def on_guild_join(guild):
    flag = True
    textchannelList = guild.text_channels
    for x in textchannelList:
        print(x)
        if(x == ChannelName):
            flag = False
        

    if(flag):
         await guild.create_text_channel(ChannelName)
   

client.run(TOKEN)


```python
import spacy
import nltk
import pandas as pd
from IPython.display import display
nlp = spacy.load("en_core_web_sm")
```

## Read focus words from File


```python
focus = []
with open("resources/focus_words.txt",'r') as fh:
    for line in fh:
        line = line.strip("\n")
        line = line.strip(' ')
        focus.append(line)
print(focus)
```

    ['kill', 'death', 'shoot', 'take', 'remove', 'kidnap', 'transport', 'train', 'fled', 'deport', 'expel', 'transfer', 'resettle', 'escape', 'run', 'murder', 'burn', 'hang', 'execute', 'throw', 'beat', 'stab', 'tuberculosis', 'epidemic', 'exterminate', 'typhus', 'dysentry', 'typhoid', 'kidnap', 'emigrate']
    


```python
with open("corpus/Blazowa.txt",'r',encoding = "utf-8") as fh:
    text = fh.readlines()
```

## Find sentences that have these focus words,

Reverted back to wordnet as the memory requirements for word_vec is huge, but will work on good computers.


```python
ls_index = []
ls_word = []
ls_match = []
for i,sent in enumerate(text):
    doc = nlp(sent)
    for token in doc:
        if token.lemma_ in focus:
            ls_index.append(i)
            ls_word.append(token.text)
            ls_match.append(token.lemma_)
d = {"SentIndex": ls_index,
       "FocusWord": ls_word,
       "MatchingWord": ls_match}
df = pd.DataFrame(d)
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SentIndex</th>
      <th>FocusWord</th>
      <th>MatchingWord</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>expel</td>
      <td>expel</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>transferred</td>
      <td>transfer</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>took</td>
      <td>take</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>running</td>
      <td>run</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>taken</td>
      <td>take</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10</td>
      <td>running</td>
      <td>run</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11</td>
      <td>taken</td>
      <td>take</td>
    </tr>
    <tr>
      <th>7</th>
      <td>16</td>
      <td>ran</td>
      <td>run</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>executed</td>
      <td>execute</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17</td>
      <td>resettled</td>
      <td>resettle</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17</td>
      <td>transport</td>
      <td>transport</td>
    </tr>
    <tr>
      <th>11</th>
      <td>18</td>
      <td>took</td>
      <td>take</td>
    </tr>
    <tr>
      <th>12</th>
      <td>18</td>
      <td>shot</td>
      <td>shoot</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19</td>
      <td>deported</td>
      <td>deport</td>
    </tr>
  </tbody>
</table>
</div>


## Extracting Chunks from each sentence.
### Chunks will later be used to enhance the triples.


```python
def get_phrase(token,sent):
    """
    Given a token that is noun or PROPN,
    get the Noun Phrase, This is a form of manual chunking
    """
    visited = set()
    visited.add(token.i)

    def visit_children(token,visited):
        if len(list(token.children)) != 0:
            for c in token.children:
                visited.add(c.i)
                visit_children(c,visited)

    visit_children(token,visited)
    visited = list(visited)
    visited.sort()
    phrase = ""
    for i in visited:
        phrase = phrase + " "+sent[i].text
    return phrase

ls_nouns = ["nsubj","dobj","pobj","nsubpass"]
ls_sentindices = []
ls_word_chunks = []
for index in df["SentIndex"]:
    sent = nlp(text[index])
    for token in sent:
        if token.dep_ in ls_nouns:
            phrase = get_phrase(token,sent)
            ls_sentindices.append(index)
            ls_word_chunks.append(phrase)
chunk_dict = {"sentIdx": ls_sentindices,
             "Chunks": ls_word_chunks}
chunk_df = pd.DataFrame(chunk_dict)
display(chunk_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentIdx</th>
      <th>Chunks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>the Soviets</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Poland</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>the east</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>September 17 , 1939</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>German troops stationed in Błażowa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>Błażowa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>as many Jews as possible</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>them</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>the San River –</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>less than 16 kilometers ( 10 miles ) which ha...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>Błażowa –</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>Soviet forces</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>A total of 57 Jews</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>57 Jews</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>Błażowa</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>768</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>July 1940</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>Łódź</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5</td>
      <td>Kalisz</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5</td>
      <td>Błażowa</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5</td>
      <td>Rzeszów</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>mid - October 1940</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5</td>
      <td>60 more</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
      <td>the same towns</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5</td>
      <td>December 1940</td>
    </tr>
    <tr>
      <th>25</th>
      <td>5</td>
      <td>the Judenrat</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5</td>
      <td>the following statistics regarding Jewish res...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>5</td>
      <td>Jewish residents in Błażowa</td>
    </tr>
    <tr>
      <th>28</th>
      <td>5</td>
      <td>Błażowa</td>
    </tr>
    <tr>
      <th>29</th>
      <td>5</td>
      <td>a total of 991 Jews , 778 ( 240 families )</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>169</th>
      <td>17</td>
      <td>Jews</td>
    </tr>
    <tr>
      <th>170</th>
      <td>17</td>
      <td>their belongings</td>
    </tr>
    <tr>
      <th>171</th>
      <td>17</td>
      <td>their wagons</td>
    </tr>
    <tr>
      <th>172</th>
      <td>17</td>
      <td>the day of the deportation to transport the g...</td>
    </tr>
    <tr>
      <th>173</th>
      <td>17</td>
      <td>the deportation to transport the ghetto resid...</td>
    </tr>
    <tr>
      <th>174</th>
      <td>17</td>
      <td>the ghetto residents</td>
    </tr>
    <tr>
      <th>175</th>
      <td>17</td>
      <td>the Rzeszów ghetto</td>
    </tr>
    <tr>
      <th>176</th>
      <td>18</td>
      <td>The liquidation of the Błażowa ghetto</td>
    </tr>
    <tr>
      <th>177</th>
      <td>18</td>
      <td>the Błażowa ghetto</td>
    </tr>
    <tr>
      <th>178</th>
      <td>18</td>
      <td>place</td>
    </tr>
    <tr>
      <th>179</th>
      <td>18</td>
      <td>June 26 , 1942</td>
    </tr>
    <tr>
      <th>180</th>
      <td>18</td>
      <td>Rzeszów</td>
    </tr>
    <tr>
      <th>181</th>
      <td>18</td>
      <td>the course of the Aktion</td>
    </tr>
    <tr>
      <th>182</th>
      <td>18</td>
      <td>the Aktion</td>
    </tr>
    <tr>
      <th>183</th>
      <td>18</td>
      <td>Jewish victims</td>
    </tr>
    <tr>
      <th>184</th>
      <td>18</td>
      <td>that day</td>
    </tr>
    <tr>
      <th>185</th>
      <td>18</td>
      <td>The liquidation of the Błażowa ghetto</td>
    </tr>
    <tr>
      <th>186</th>
      <td>18</td>
      <td>the Błażowa ghetto</td>
    </tr>
    <tr>
      <th>187</th>
      <td>18</td>
      <td>place</td>
    </tr>
    <tr>
      <th>188</th>
      <td>18</td>
      <td>June 26 , 1942</td>
    </tr>
    <tr>
      <th>189</th>
      <td>18</td>
      <td>Rzeszów</td>
    </tr>
    <tr>
      <th>190</th>
      <td>18</td>
      <td>the course of the Aktion</td>
    </tr>
    <tr>
      <th>191</th>
      <td>18</td>
      <td>the Aktion</td>
    </tr>
    <tr>
      <th>192</th>
      <td>18</td>
      <td>Jewish victims</td>
    </tr>
    <tr>
      <th>193</th>
      <td>18</td>
      <td>that day</td>
    </tr>
    <tr>
      <th>194</th>
      <td>19</td>
      <td>July 1942</td>
    </tr>
    <tr>
      <th>195</th>
      <td>19</td>
      <td>Błażowa</td>
    </tr>
    <tr>
      <th>196</th>
      <td>19</td>
      <td>the other Jews concentrated in the Rzeszów gh...</td>
    </tr>
    <tr>
      <th>197</th>
      <td>19</td>
      <td>the Rzeszów ghetto</td>
    </tr>
    <tr>
      <th>198</th>
      <td>19</td>
      <td>the Bełżec extermination camp</td>
    </tr>
  </tbody>
</table>
<p>199 rows × 2 columns</p>
</div>


## Triple extraction logic.

### Reverted back to original simple logic as the complex one has too many bugs.

Since we went back to this simple logic of triple extraction. We are no longer able to generate all potential triples.
The set of triple obtained is a small subset of all triples that actually exist in the text.


```python
def get_triple(sent):
    nouns = list(sent.noun_chunks)
    for token in sent:
        if token.dep_ in ["nsubj","nsubjpass"] and token.head.pos_ == "VERB":
            vphrase = token.head
            sphrase = token
            for possible_object in vphrase.children:
                if possible_object.dep_ in["dobj","iobj","pobj"]:
                    return(sphrase,vphrase,possible_object)
```

### Chunks from the previous step is used here to obtain triples that are descriptive.

The triples obtained here are not reflective of the intended results.


```python
count = 0
total = 0
ls_sent_indices = []
ls_subjects = []
ls_verbs = []
ls_objects = []

def get_chunk(sentidx,word):
    match = []
    for i,chunk in zip(ls_sentindices,ls_word_chunks):
        if i == sentidx:
            match.append(chunk)
        for w in match:
            if word.text in w:
                return w
        return word.text

for i,line in enumerate(text):
    sent = nlp(line)
    tup= get_triple(sent)
    if not(tup is None):
        s,v,o = tup
        subj = get_chunk(i,s)
        obj = get_chunk(i,o)
        ls_sent_indices.append(i)
        ls_subjects.append(subj)
        ls_verbs.append(v)
        ls_objects.append(obj)
d = {"SentIndx": ls_sent_indices,
     "Subjects": ls_subjects,
    "Verbs: ": ls_verbs,
    "Objects: ": ls_objects}
df = pd.DataFrame(d)
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SentIndx</th>
      <th>Subjects</th>
      <th>Verbs:</th>
      <th>Objects:</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>﻿Błażowa</td>
      <td>located</td>
      <td>kilometers</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>the Soviets</td>
      <td>invaded</td>
      <td>Poland</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>authorities</td>
      <td>set</td>
      <td>council</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Poles</td>
      <td>worked</td>
      <td>estate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Germans</td>
      <td>invaded</td>
      <td>Union</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>they</td>
      <td>had</td>
      <td>relatives</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Judenrat</td>
      <td>opened</td>
      <td>kitchen</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Natansohn</td>
      <td>chaired</td>
      <td>committee</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>it</td>
      <td>served</td>
      <td>children</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>craftsmen</td>
      <td>running</td>
      <td>businesses</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>They</td>
      <td>received</td>
      <td>board</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>Reich</td>
      <td>requested</td>
      <td>man</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>Jews</td>
      <td>given</td>
      <td>days</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>establishment</td>
      <td>influenced</td>
      <td>’s</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>who</td>
      <td>had</td>
      <td>jobs</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>authorities</td>
      <td>permitted</td>
      <td>market</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18</td>
      <td>liquidation</td>
      <td>took</td>
      <td>place</td>
    </tr>
  </tbody>
</table>
</div>


## Extracting Meta-data.

Verbs that are part of our extracted triples may contain some addition information. Based on observations this additional information is usually attached to the preposition of the verb. 

This Code can determine such prepositions and the verbs they are attached to to extract meta-data for the verb.

This meta data can be also encoded as triple as shown below in results.


```python
ls_sindex = []
ls_verb = []
ls_prep = []
ls_data = []

for index,line in enumerate(text):
    sent = nlp(line)
    for token in sent:
        if token.dep_ == "prep" and token.head.pos_ == "VERB" and token.head.lemma_ in focus:
            ls_sindex.append(index)
            ls_verb.append(token.head.text)
            ls_prep.append(token.text)
            word = get_phrase(list(token.children)[0],sent)
            ls_data.append(word)
meta_dict = {"sentIndex": ls_sindex,
             "Verb": ls_verb,
            "preposition": ls_prep,
            "Meta": ls_data}
meta_df = pd.DataFrame(meta_dict)
display(meta_df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentIndex</th>
      <th>Verb</th>
      <th>preposition</th>
      <th>Meta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>expel</td>
      <td>by</td>
      <td>ordering them to cross the San River –</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>transferred</td>
      <td>By</td>
      <td>July 1940</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>transferred</td>
      <td>to</td>
      <td>Błażowa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>transferred</td>
      <td>via</td>
      <td>Rzeszów</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>running</td>
      <td>In</td>
      <td>July 1941</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11</td>
      <td>taken</td>
      <td>In</td>
      <td>mid - June 1941</td>
    </tr>
    <tr>
      <th>6</th>
      <td>11</td>
      <td>taken</td>
      <td>to</td>
      <td>Rzeszów</td>
    </tr>
    <tr>
      <th>7</th>
      <td>16</td>
      <td>ran</td>
      <td>in</td>
      <td>panic</td>
    </tr>
    <tr>
      <th>8</th>
      <td>17</td>
      <td>resettled</td>
      <td>In</td>
      <td>May or June 1942</td>
    </tr>
    <tr>
      <th>9</th>
      <td>17</td>
      <td>resettled</td>
      <td>into</td>
      <td>the Błażowa ghetto</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17</td>
      <td>transport</td>
      <td>to</td>
      <td>the Rzeszów ghetto</td>
    </tr>
    <tr>
      <th>11</th>
      <td>18</td>
      <td>took</td>
      <td>on</td>
      <td>June 26 , 1942</td>
    </tr>
    <tr>
      <th>12</th>
      <td>18</td>
      <td>shot</td>
      <td>in</td>
      <td>the course of the Aktion</td>
    </tr>
    <tr>
      <th>13</th>
      <td>19</td>
      <td>deported</td>
      <td>In</td>
      <td>July 1942</td>
    </tr>
    <tr>
      <th>14</th>
      <td>19</td>
      <td>deported</td>
      <td>to</td>
      <td>the Bełżec extermination camp</td>
    </tr>
  </tbody>
</table>
</div>


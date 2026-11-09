# Project Title & Overview 
Federalist papers according to Britannica was a series of 85 published essays on the proposed new Constitution of the United States. While the Papers all appeared with the signature of 'PUBLIUS', the authorship of some of the papers were disputed as there was a difference with the lists of authorship claimed by Alexander Hamilton and James Madison. However with computer analysis and historical evidence, scholarly consensus was made wherein: Hamilton wrote: 1, 6-9, 11-13, 15-17, 21-36, 59-61, and 65-85. Madison wrote: 10-14, 18-20, 37-58, and 62-63. Jay wrote: 2-5 and 64. Wherein 18-20 was apparently a joint effort between Hamilton and Madison. As such this project attempts to reproduce/refute on its own with just authorship attribution methods the results these consensus has done without intentionally replicating the methods.   
# Research Question & Objectives 
The research question this project seeks to ask is: 
The objectives are: 

# Dataset  
The Dataset was sourced from Project Gutenberg's Federalist Papers as found in the link below: https://www.gutenberg.org/ebooks/1404 

The file itself was downloaded as the text variant and was parsed and classified into 4 different classes: The Disputed Papers, Jay's Papers, Accepted Papers, and the Joint Papers. 
The class chosen to be explored were the accepted papers as Jay's papers weren't the focal point of the dispute due to the nature of his role in the creation of the Federalist Papers. 
# Methodology 
## Text Preprocessing 
All of the text preprocessing was done with the usage of spacy which is a free open-sourced library for Natural Language Processing. This was done with the use of spacy's pipe 
- Tokenization - Each of the paper was used as an instance and for each of the papers words were segmented. 
- Lower - Each words were lowered using ```.lower()``` to easily find duplicates and proper counts
- Normalization - All text were normalized with tfidfvectorizer's 'l2' 
## Feature Engineering
- Features were engineered based on some of the most common indicators of authorship: Lexical Features/Stylistic Features and such. 
- Features were separated into two types: Vector and Scalar Features. Vector Features were features such as: the most common words, parts of speech, the most common suffixes(3 word ngram), word chunks(bigrams - 2 word). Scalar features on the other hand were stopword rates, punctuation rates and unique word rates. 
## Feature Selection  
Table ablation here 
- While the style feature set scored higher than function words, the differences weren't significant and so the researcher chose to utilize the feature set with the least features. 
## Performance Metrics
For Performance: A confusion matrix was employed, along with the f1 mean, f1 standard deviation, recall, and precision. These were done as the final test employed the usage of LOOCV due to limited samples and as such means were needed to aggregate. 
## Results & Discussion
Heatmap table here 
Stop rate and unique rates, punct rates and length were all strongly correlated, while the unique rate and length, punct rate and stoprate, were all negatively correlated. This likely means that most of the stylistic features are all somewhat length extensions as the higher the length of text, they either exhibit less unique features, higher use of stopwords, and higher use of punctuations. 
Finalmodel on an LOOCV split table here (CF Matrix) + Metrics 
As we can see 
Most impactful features here: 
Basically Hamilton used words such as 'upon', 'there', 'to' and other words, while Madison used words such as 'on', 'by', 'and' etc. 

Disputed and Joint Papers table here 
All of the Disputed Papers except for Paper No. 55 were classified as Madison's with high Probabilities (0.802-0.998). While Paper no. 55 was classified as 'Uncertain' based on a simple metric(> 0.6 = Madison and 0.4 < = Hamilton and 0.4 - 0.6 = Uncertain) it matches Mollester's wallace claim of Paper No. 55's classification between the 2 as the weakest.  
As for the Joint Papers they were all overwhelmingly classified as Madison's (0.940, 0.957, 0.993) 
# Installation & Usage 
1. Install the requirements:  ```pip install -r requirements.txt``` & ```python -m spacy download en_core_web_md```
2. And run the notebook as is. If the papers weren't recognize simply uncomment and run Cell
# Limitations & Future Improvements 
This research has only implemented the most basic stylometric features to determine authorship. While it was found sufficient, better features engineered could provide a better separation of classes and could find some patterns specially co-authorship in the Joint papers. 
# Reference & Acknowledgements 
1. Mollester and Wallace 
2. A Survey of Modern Authorship Attribution Methods - Efstathios Stamatatos

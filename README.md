
# Feature engineering with logistic regression to classify movie reviews

*Submission for University of Arizona Spring 2024 Statistical NLP Class Kaggle Competition*

*Alison Kleczewski*
<br /><br />

## Task
The dataset for this project contains reviews which fall into 3 categories:

- Not a movie or TV show review (label 0)
- Positive movie or TV show review (label 1)
- Negative movie or TV show review (label 2)

There are two files in CSV format--one containing training data and one containing test data. The training file contains three columns: ID, review, and label. The test file contains two columns: ID and review.  The goal was to develop and train a classification model using the labeled data from the training file, in order to then accurately predict which of the three categories each test review should fall into.
<br /><br />
## Approach

For my first attempt, I created my own custom features and trained a logistic regression model on them. First, I preprocessed the data by replacing NaN items with an empty string, removing certain punctuation I thought would not be relevant for sentiment, making everything lowercase, and getting rid of newline characters and HTML line break tags. Then I came up with some features I felt would signal the categories of the reviews. I used DictVectorizer to transform them into matrices, and then trained a logistic regression model on them. I created various custom features along these lines:

```Python
{
'sentiment_score':  sentiment_score, #sentiment score from NLTK SentimentIntensityAnalyzer
'exclamation_count': text.count('!'),
'review_length': len(text.split()),
'movie_tv_words': 1  if re.search(r'(movie|film|episode|watch)', text) else  0, 
'non_movie_words': 1  if re.search(r'(book|author|novel|read|composer)', text) else  0, 
'contains_positive': 1  if re.search(r'(incredible|brilliant|masterpiece|excellent|good|great|love)', text) else  0, 
'contains_negative': 1  if re.search(r'(dull|boring|stupid|terrible|awful|dumb|horrible|bad|waste|yawn|hated)', text) else  0
}
```

The goal was a customizable setup in which I could define and control the features precisely. However, based on my relatively small set of custom features in this draft setup, my initial Kaggle submission resulted in an accuracy score of around 77% percent. While I still think this approach could be promising with enough custom features, it would have been extremely time-consuming and manual. The training set has about 70,000 reviews, so coming up with a strong and comprehensive list of custom features to cover the many different ways of expressing sentiment would have been difficult. I still like the idea behind this approach, since it offers the ability to carefully control the features. If I were to continue with it, I would develop a larger set of features, and possibly consider alternatives like a count-based method instead of a binary one. However, I decided to shift gears and try a model involving less manual work.

After considering more efficient methods to come up with features, I settled on using scikit-learn's TF-IDF vectorizer. This is a useful strategy for various reasons:

 - It can be done automatically by sci-kit learn using different ngram counts (no need to come up with features manually)
 - It is considered a good "go to" baseline (Jurafsky & Martin, [_Speech and Language Processing_](https://web.stanford.edu/~jurafsky/slp3/ed3bookfeb3_2024.pdf), p. 117)
 - An advantage of TF-IDF (Term Frequency-Inverse Document Frequency) over raw counts is that TF-IDF reduces the importance of words that appear frequently across all documents (and are therefore unlikely to help distinguish classes). TF-IDF not only effectively filters out stop words but also diminishes the impact of other non-discriminative words. This is particularly useful given the large number of reviews in the dataset--it would take a long time to sift through and try to figure out which words do not help differentiate between classes.

Having settled on TF-IDF for feature extraction, I decided I would continue using logistic regression for training, since this worked well for a previous spam classifier experiment I tested, and would integrate easily with TF-IDF. I also refined my code in other ways to ensure a more successful model. For my second attempt, I implemented an 80%/20% (training and development) split of the data to help catch issues like overfitting and to test how my model would generalize to unseen data.

I set up the new model with some basic parameters, sticking mostly with the defaults as a starting point. I chose to start with a 1-3 ngram range as I thought this might capture at least some types of negation and more complex sentiment phrases (e.g. "do not watch" / "good acting but"). Right away, this model was a big improvement over the previous one, giving me a weighted F1 score on my development data of around 90%, so I decided to stick with this approach and further refine it.

As a next step, I experimented with different parameters and explored other ways to improve my F1 score. I explored more preprocessing steps like removing stop words, lemmatizing (using NLTK WordNetLemmatizer), and replacing words like "film", "flick", and "movies" with "movie" (and doing something similar for TV show-related words). Interestingly, these various preprocessing steps tended to lower my F1 score by a marginal amount (around 0.4% - 0.5% depending on various factors) as compared to the default tokenizer and preprocessor steps in scikit-learn TF-IDF. According to the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), the default token pattern used is  _'(?u)\b\w\w+\b'_ and punctuation is ignored and treated as a separator. I decided to stick with these defaults since my additional preprocessing steps did not improve my F1 score and even lowered it in some cases. It is possible the stop word removal did not help because TF-IDF essentially takes care of this, or because the particular words being removed could be meaningful for sentiment analysis (e.g. negation terms like "not"). I would guess that the lack of improvement using lemmas could have to do with the reviews having a lot of informal language and slang which are not effectively lemmatized. It was unclear to me why having more variation of terms like "movie", "film", etc. made for a better model, which I'd like to investigate further at some point.

Another thing I noticed while testing my model, was that the unigram and bigram "br" and "br br" (from HTML line break tags) would come in as top coefficients for classes 1 and 2 if not removed during preprocessing. I had thought these tokens would act as noise that would negatively impact the accuracy, but when I used a token pattern or preprocessor to remove those, it actually made the F1 score marginally worse (by about 0.35%, depending on the version of my model). This seemed counterintuitive and I'm still unsure about the cause. It's possible the line breaks occur in certain types of reviews more often and might actually correlate with the classes in a meaningful way. Otherwise, it could be that my model is overfitting to the training data and leaving the "br" features will work against me in the final test run. However, I chose to leave them given the higher F1 score for my development data.

After settling on the scikit-learn preprocessing defaults, the next step was fine-tune the parameters to see how much I could improve my accuracy and found that an efficient way to do this would be to use scikit-learn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) method (explained further [here](https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/#h-what-is-gridsearchcv)). This presented some new challenges because many combinations of parameters were not feasible to test in a reasonable amount of time. However, it proved to be useful overall if I limited my search to just a few select parameter comparisons.

After a lot of experimenting, I started to zero in on some effective parameters. For TF-IDF, I found that a minimum document frequency of 3 and a maximum document frequency of 70% worked well--limiting the most rare and most common ngrams seemed to help the model focus on what was important. Logarithmic scaling with "sublinear_tf=True" improved the model, as did lowercasing, using a binary representation, and leaving the default L2 norm. With respect to ngrams, I found that ranges between 1-3 and 1-5 were most accurate for my model. Smaller ranges likely did not capture enough complexity and larger ranges probably led to overfitting. For logistic regression parameters, I found that regularization with a C value between 5 to 20 worked best (depending on other various parameters). It was interesting to see how changing the C value impacted the training and development accuracy. Increasing C gave me increasingly good accuracy results for the training data, but at a certain point the development data would stop improving, signaling that the model was overfitting. Through various adjustments and testing with GridSearchCV, I also found that the Stochastic Average Gradient Augmented ('saga') solver, a balanced class weight, and the default L2 penalty worked well for my model.

After I felt I had reached a peak in terms of significant F1 improvements from adjusting TF-IDF and logistic regression parameters, I was curious about what other methods could improve my score. In researching techniques, I came across scikit-learn's "[SelectKBest](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)" method (discussed further [here](https://medium.com/@Kavya2099/optimizing-performance-selectkbest-for-efficient-feature-selection-in-machine-learning-3b635905ed48#098d)) . Using statistical tests to measure dependence between variables (i.e. features as independent variables and classes as dependent variables), SelectKBest can help identify the most relevant features from the data, which can improve the performance of the model and reduce overfitting. My TF-IDF model with 1-5 ngrams originally contained 1,050,400 features! After playing around with different K values using the chi-squared test, I found that 215,000 seemed to be the sweet spot for improving my F1 score. With SelectKBest implemented, I was able to get an F1 score around 0.9458 for my development data. While it felt like I could have fiddled around endlessly, testing more feature extraction methods, training techniques, and parameters, I decided this F1 score was satisfactory.

Here is the final approach I ended up with:
- Load the data using Pandas and replace missing values with an empty string using the "fillna" function
- Use TF-IDF Vectorizer from scikit-learn to convert text data into TF-IDF features
- Use SelectKBest with the chi-squared test to select the top K most significant features
- Train a logistic regression model on these features and use it to make predictions on the test data

*See GitHub link under the "Code" section below to see the specific parameters used.

<br /><br />
## Results

With my final approach of TF-IDF for feature extraction, pruning with SelectKBest, and using logistic regression for training, I ended up with an F1 score of 0.9458 for my development data. This translated to a 0.92526 F1 score on Kaggle for the initial run with 80% of the test data, and it will be interesting to see how much it changes with the remaining 20% of the test data.

The confusion matrix for my development data gave the following results:

|            | Predicted 0 | Predicted 1 | Predicted 2 |
|------------|-------------|-------------|-------------|
| **Actual 0** | 6369        | 47          | 38          |
| **Actual 1** | 112         | 3523        | 221         |
| **Actual 2** | 61          | 281         | 3412        |

F1 scores per class were as follows:
-   **Class 0**: 0.9801
-   **Class 1**: 0.9142
-   **Class 2**: 0.9191

<br /><br />
## Code

You can find my code here: [https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-aliklec](https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-aliklec)

**How to run the code:**
1. Download or clone the Jupyter notebook "ReviewClassifier.ipynb" from the [GitHub link](https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-aliklec).
2. Download the "train.csv" and "test.csv" files from [Kaggle](https://www.kaggle.com/competitions/ling-539-sp-2024-class-competition/data).
3. Make sure "train.csv" and "test.csv" are in the same directory as the Jupyter notebook.
4. Confirm that you have Pandas, NumPy, and Scikit-learn installed (at the beginning of the Jupyter notebook, you’ll find three commented lines that can be uncommented for installing these libraries via pip, if needed).
5. Open the Jupyter notebook "ReviewClassifier.ipynb" and navigate to "Kernel" --> "Restart & Run All" to run it. The test predictions will be printed to a file called "results.csv" in the same directory.

*Note: the last few cells in the notebook are commented out. These include cells to print things like misclassifications, a confusion matrix, and GridSearchCV best parameters. I have kept them in the notebook for possible future development of the model.*

**Python Version/Dependencies**

The code was developed using the following versions:

`Python==3.11.5`

`Pandas==2.0.3`  

`NumPy==1.24.3`  

`Scikit-learn==1.3.0`

*Using a different version of Python or these libraries may change the code's behavior.

<br /><br />
## Challenges/Ideas for Future Improvement

The review dataset presented many interesting challenges that could be explored further, and that my model did not address. As mentioned earlier, I couldn't determine why maintaining the "br" tags improved my model's accuracy, and I would like to understand this better. It was also a mystery to me why replacing terms like "movie", "film", and "flick" with a single, unified version decreased the accuracy. These are some patterns I would like to understand better. 

I would also like to investigate other preprocessing steps and how they could impact the results. For example, many of the reviews had typos or informal spellings that could confuse the analysis. Some examples from the data are things like "surprized" and "MOVIE SUUUUUUUUUUUUCKS"---ideally these would be normalized to "surprised" and "movie sucks" to enhance feature alignment. Another interesting aspect of the data was the inclusion of languages other than English. This may have added variability and confusion to the dataset. Nonetheless, the TF-IDF model seems capable of handling multiple languages to some extent. For instance, when examining the top features, I noticed some Chinese characters, indicating that the TF-IDF model might have identified important terms from non-English reviews that suggest specific categories. However, these non-English reviews were a smaller part of the dataset and may not have constituted enough data for the model to pick up on many useful patterns. Ideally, for future improvements of the classifier, all reviews would be translated into English during preprocessing to standardize the data.

Another challenge highlighted by the confusion matrix was the model mistaking positive reviews for negative ones, and vice versa. In reviewing the top coefficients, common terms such as "film" and "movie" were identified as key features for both classes, which likely led to mislabeling. A potentially better approach could be two-stage classification: initially, the model would be trained to differentiate between TV/movie reviews and non-TV/movie reviews. Once TV/movie reviews are isolated, the model could then focus on classifying them as positive or negative. This method could also address situations such as the word "worst" being strongly associated with negative reviews during training, which might tip the classification of a non-TV/movie review into the negative category.

Perhaps the biggest challenge is that the sentiment in many reviews is complex, often blending emotions or using verbal irony in ways that aren't easily classified as "positive" or "negative"—even for a human reviewer. For example:

> ***I got lost a hundred times but didn't mind, because the movie is so bad, it's real fun to watch. Zero-Budget trash with actors not deserving that name. Go check it out!*** 
(True label: 1 , Predicted: 2)



> ***This was the worst movie I've ever seen, yet it was also the best movie. Sci Fi original movie's are supposed to be bad, that's what makes them fun!...So go watch this movie, but don't buy it.*** (True label: 1, Predicted: 2)



> ***This is the best movie I've seen since I got the scope at the proctologists office...this morning.*** (True label: 2, Predicted: 1)



There were many examples like this of reviewers using sarcasm or expressing both likes and dislikes in the same review, leading to a complex mix of sentiments. These subtle uses of language present a challenge for a basic TF-IDF model, which analyzes word frequency and lacks the ability to interpret the context to understand sarcasm or mixed feelings.  These are the interesting language complexities that a simple TF-IDF model cannot grasp. This limitation highlights the need for more sophisticated models that can better grasp the nuances of human communication.

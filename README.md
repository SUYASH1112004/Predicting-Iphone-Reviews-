# Predicting-Iphone-Reviews-
created a project that analyzed customer reviews of iPhones using Natural Language Processing (NLP) and Logistic Regression.

.The model predicts based on the reviews of Iphone customers and the ratings they have given .

 Here’s a quick overview of the process:

 Steps Involved:

 1] Loaded and preprocessed the dataset to handle missing values and created a binary sentiment label (positive vs. negative).

2] Visualized the data distribution to identify imbalances in ratings.

3] Applied TF-IDF Vectorization to extract meaningful features from the text.

4] Built a Logistic Regression Model with balanced class weights to handle the skewed data.

5] Evaluated the model using metrics like AUC (Area Under Curve) to ensure accuracy.



Key Insights:

The most influential words in determining sentiment were identified, revealing what customers value most in their iPhone experience.

The model learns from the star rating (1-start , 5-star) and text review of customer .The text reviews are transformed in  numeric feature by TFIDF vectorizer.

Real-time prediction is also implemented to classify new reviews as positive or negative.



 This project reinforced the power of NLP and Machine Learning in turning unstructured data into actionable insights. It’s amazing how a few lines of code can extract meaningful patterns from large datasets!



 Tech Stack:

Libraries: pandas, numpy, seaborn, sklearn, matplotlib

Techniques: TF-IDF, Logistic Regression, and Class Imbalance Handling


import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class EmailDetector:
    def __init__(self, model_path='app/models/mail/spam_classifier_lr.pkl', vect_path='app/models/mail/count_vectorizer.pkl'):
        """
        Initialize the EmailDetector by loading the trained model and vectorizer.
        """
        self.lr = joblib.load(model_path)
        self.cVect = joblib.load(vect_path)
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Tokenize and remove stopwords from the input text.
        """
        return ' '.join([word for word in word_tokenize(text) if word.lower() not in self.stop_words])

    def predict(self, text):
        """
        Predict whether the input text is Spam or Not Spam.
        Returns prediction and probability dictionary.
        """
        processed_text = [self.preprocess_text(text)]
        t_dtv = self.cVect.transform(processed_text).toarray()
        
        pred_class = 'Spam' if self.lr.predict(t_dtv)[0] else 'Not Spam'
        prob = self.lr.predict_proba(t_dtv)[0]
        prob_dict = {'Not Spam': round(prob[0]*100, 2), 'Spam': round(prob[1]*100, 2)}
        
        return pred_class, prob_dict


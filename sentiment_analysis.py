from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def detect_sentiment(user_input):
    """Analyzes sentiment of user input (Positive, Neutral, Negative)."""
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(user_input)["compound"]
    
    if sentiment_score > 0.5:
        return "Positive"
    elif sentiment_score < -0.5:
        return "Negative"
    else:
        return "Neutral"

if __name__ == "__main__":
    test_input = "I feel really anxious and sad today."
    sentiment = detect_sentiment(test_input)
    print("Sentiment Detected:", sentiment)

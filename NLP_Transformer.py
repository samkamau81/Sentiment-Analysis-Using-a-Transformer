import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def classify_tweet(tweet):
    input_ids = tokenizer.encode(tweet, add_special_tokens=True, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    predicted_sentiment = torch.argmax(logits, dim=1).item()
    return 'positive' if predicted_sentiment == 1 else 'negative'


tweet = "I just saw the new Marvel movie and it was amazing!"
sentiment = classify_tweet(tweet)
print(sentiment)

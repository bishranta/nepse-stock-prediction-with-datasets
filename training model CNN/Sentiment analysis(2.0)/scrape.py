from bs4 import BeautifulSoup
import requests
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

# Load tokenizer and model once (outside the loop)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, framework="tf")

def pipelineMethod(payload):
    return classifier(payload)[0]  # Directly return the sentiment result

# DataFrame to store results
column = ['datetime', 'title', 'source', 'link', 'top_sentiment', 'sentiment_score']
df = pd.DataFrame(columns=column)

counter = 0
for page in range(1, 41):
    url = f'https://www.investopaper.com/articles/page/{page}/#wpnw-news-{page - 1}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')

    articles = soup.find_all('div', class_='news-content')
    print(f"Found {len(articles)} articles on page {page}")

    for article in articles:
        # Extract date
        date_post = article.find('div', class_='grid-date-post')
        datetime = date_post.text.strip().split('/')[0] if date_post else 'N/A'

        # Extract title and link
        title_tag = article.find('h3', class_='news-title')
        title = title_tag.find('a').text.strip() if title_tag and title_tag.find('a') else 'N/A'
        link = title_tag.find('a').get('href') if title_tag and title_tag.find('a') else 'N/A'

        # Extract source
        source_tag = date_post.find_all('a') if date_post else []
        source = source_tag[1].text.strip() if len(source_tag) > 1 else 'N/A'

        # Get sentiment analysis
        output = pipelineMethod(title)
        top_sentiment = output['label']
        sentiment_score = output['score']

        # Append to DataFrame
        df = pd.concat([pd.DataFrame([[datetime, title, source, link, top_sentiment, sentiment_score]],
                                     columns=df.columns), df], ignore_index=True)

        counter += 1

print(f'\n{counter} news articles scraped')
df.to_csv('sentiment-nepali-news.csv', index=False)

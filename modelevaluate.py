# Plot class distribution
sns.countplot(data=data, x='class', order=data['class'].value_counts().index)
plt.title("Class Distribution")
plt.show()

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(
            token.lower() for token in sentence.split()
            if token.lower() not in stop_words
        ))
    return preprocessed_text

# Apply preprocessing
data['text'] = preprocess_text(data['text'].values)

# WordCloud for Real news
real_text = ' '.join(word for word in data['text'][data['class'] == 1])
wordcloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False).generate(real_text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Real News")
plt.show()

# WordCloud for Fake news
fake_text = ' '.join(word for word in data['text'][data['class'] == 0])
wordcloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110, collocations=False).generate(fake_text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Fake News")
plt.show()


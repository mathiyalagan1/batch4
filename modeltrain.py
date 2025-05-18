def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# Bar chart of top words
common_words = get_top_n_words(data['text'], 20)
df1 = pd.DataFrame(common_words, columns=['Word', 'Count'])
df1.groupby('Word').sum()['Count'].sort_values(ascending=False).plot(
    kind='bar', figsize=(10, 6), xlabel="Top Words", ylabel="Count", title="Top Words Frequency"
)
plt.show()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
print("Logistic Regression - Train Accuracy:", accuracy_score(y_train, lr_model.predict(x_train)))
print("Logistic Regression - Test Accuracy:", accuracy_score(y_test, lr_model.predict(x_test)))

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
print("Decision Tree - Train Accuracy:", accuracy_score(y_train, dt_model.predict(x_train)))
print("Decision Tree - Test Accuracy:", accuracy_score(y_test, dt_model.predict(x_test)))

# Confusion Matrix
cm = metrics.confusion_matrix(y_test, dt_model.predict(x_test))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

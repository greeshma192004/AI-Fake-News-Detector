import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Get input
news = input("Enter News: ")

# Convert text
news_vector = vectorizer.transform([news])

# Prediction
prob = model.predict_proba(news_vector)

print("\nConfidence:")
print(prob)

prediction = model.predict(news_vector)

if prediction[0] == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")

# Confidence
prob = model.predict_proba(news_vector)

fake_prob = prob[0][0] * 100
real_prob = prob[0][1] * 100

print(f"\nFake Probability: {fake_prob:.2f}%")
print(f"Real Probability: {real_prob:.2f}%")

if prediction[0] == 1:
    print("\nREAL NEWS")
else:
    print("\nFAKE NEWS")

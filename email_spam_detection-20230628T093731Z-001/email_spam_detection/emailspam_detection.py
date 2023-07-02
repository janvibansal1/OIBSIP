#EMAIL SPAM DETECTION#
#BY JANVI BANSAL
import pandas as pd


df = pd.read_csv('spam.csv', encoding='latin-1')

# Remove any unnecessary columns
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# Rename columns to more meaningful names
df.columns = ['label', 'message']

# Convert label to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2)

# Vectorize the messages
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)


from sklearn.naive_bayes import MultinomialNB


clf = MultinomialNB()
clf.fit(X_train_counts, y_train)


X_test_counts = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_counts)

# Add predicted labels to the test set DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['predicted_label'])
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
test_df = pd.concat([X_test, y_test, y_pred_df], axis=1)


spam_messages = test_df[test_df['predicted_label'] == 1]['message']
print("The following messages were classified as spam:")
for i, message in enumerate(spam_messages):
    print(f"{i}. {message}")
    

index = input("Enter the index of a message to mark as not spam (or press enter to skip): ")
while index:
    try:
        index = int(index)
        if index < 0 or index >= len(spam_messages):
            raise ValueError()
        else:
            
            spam_messages.drop(spam_messages.index[index], inplace=True)
            print(f"Message {index} marked as not spam.")
            index = input("Enter the index of another message to mark as not spam (or press enter to skip): ")
    except ValueError:
        index = input("Invalid input. Please enter a valid index (or press enter to skip): ")




#Thank you!
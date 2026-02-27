import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("library_data.csv")

# Convert date columns
data['Issue_Date'] = pd.to_datetime(data['Issue_Date'])
data['Return_Date'] = pd.to_datetime(data['Return_Date'])

# Create Late_Days column
data['Late_Days'] = (data['Return_Date'] - data['Issue_Date']).dt.days - 10
data['Late_Days'] = data['Late_Days'].apply(lambda x: x if x > 0 else 0)

print("\n===== DATASET =====")
print(data)

# ---------------- ANALYSIS ----------------
print("\nMost Borrowed Books:")
print(data['Book_Name'].value_counts())

# Plot
data['Book_Name'].value_counts().plot(kind='bar')
plt.title("Most Borrowed Books")
plt.xlabel("Book Name")
plt.ylabel("Count")
plt.show()

# Department-wise usage
data.groupby('Department')['Book_Name'].count().plot(kind='pie', autopct='%1.1f%%')
plt.title("Department Wise Library Usage")
plt.show()

# ---------------- RECOMMENDATION SYSTEM ----------------

encoder = LabelEncoder()
data['Book_ID'] = encoder.fit_transform(data['Book_Name'])

user_book_matrix = pd.crosstab(data['Student_ID'], data['Book_ID'])

similarity = cosine_similarity(user_book_matrix)

def recommend_books(student_id):
    idx = user_book_matrix.index.tolist().index(student_id)
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    similar_students = [user_book_matrix.index[i[0]] for i in scores[1:3]]

    books = data[data['Student_ID'].isin(similar_students)]['Book_Name'].unique()
    return books

print("\nRecommended books for Student 101:")
print(recommend_books(101))

# ---------------- LATE RETURN PREDICTION ----------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data['Late'] = data['Late_Days'].apply(lambda x: 1 if x > 0 else 0)

X = data[['Late_Days']]
y = data['Late']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nLate Return Prediction Accuracy:", accuracy)

# Predict new case
new_data = [[5]]
prediction = model.predict(new_data)
print("Prediction for 5 late days (1 = Late, 0 = On time):", prediction)
